"""
Convert BIDCell segmentation mask → Xenium Explorer-compatible parquet files.
Outputs written to outs_subset/bidcell_segmentation/.

Usage:
  conda run -n bidcell python bidcell_to_xenium.py
"""

import numpy as np
import pandas as pd
import tifffile
import pyarrow as pa
import pyarrow.parquet as pq
import glob, os
from pathlib import Path
from skimage.measure import find_contours, regionprops

# ── Paths ──────────────────────────────────────────────────────────────────────
MASK_PATH = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/outs_subset/model_outputs"
                 "/2026_04_02_15_55_58/test_output/epoch_1_step_4000.tif")
TX_PATH   = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/outs_subset/transcripts.parquet")
OUT_DIR   = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/outs_subset/bidcell_segmentation")
OUT_DIR.mkdir(exist_ok=True)

# ── Coordinate offset: subset → original Xenium space ─────────────────────────
# The subsetting script subtracted x_min_um and y_min_um. Add them back here
# so polygons align when opening the full outs/ dataset in Xenium Explorer.
COORDS_DIR = "/data-hdd0/Ethan_Baird/Dec25_xenium/selection_coordinates_xenium_explorer"
xs_all, ys_all = [], []
for fp in sorted(glob.glob(os.path.join(COORDS_DIR, "*.csv"))):
    for line in open(fp, encoding="latin-1").readlines()[3:]:
        line = line.strip().strip('"')
        if "," in line:
            x, y = line.split(",")
            xs_all.append(float(x)); ys_all.append(float(y))
MARGIN_UM = 200
X_OFFSET  = max(0.0, min(xs_all) - MARGIN_UM)
Y_OFFSET  = max(0.0, min(ys_all) - MARGIN_UM)
print(f"Coordinate offset to original space: +{X_OFFSET:.2f} µm x, +{Y_OFFSET:.2f} µm y")

# ── Load mask ──────────────────────────────────────────────────────────────────
print("Loading BIDCell segmentation mask...")
mask = tifffile.imread(MASK_PATH)   # (H, W) uint32
H, W = mask.shape
print(f"  Shape: {mask.shape}, cells: {mask.max()}")

# ── Extract polygon boundaries (efficient: per-cell bounding box) ──────────────
print("Extracting cell boundaries...")
props = regionprops(mask)

cell_id_col, vx_col, vy_col, lid_col = [], [], [], []
cell_rows = []

for prop in props:
    label   = prop.label
    cell_id = f"bidcell-{label}"

    # Work only within the cell's bounding box + 1px padding
    r0, c0, r1, c1 = prop.bbox
    r0p = max(0, r0 - 1);  c0p = max(0, c0 - 1)
    r1p = min(H, r1 + 1);  c1p = min(W, c1 + 1)

    sub = (mask[r0p:r1p, c0p:c1p] == label).astype(np.uint8)
    contours = find_contours(sub, level=0.5)
    if not contours:
        continue

    contour = contours[np.argmax([len(c) for c in contours])]

    # Downsample to max 300 vertices
    if len(contour) > 300:
        idx = np.round(np.linspace(0, len(contour) - 1, 300)).astype(int)
        contour = contour[idx]

    # Convert sub-image coords → full-image µm + original-dataset offset
    vx = (contour[:, 1] + c0p).astype(np.float32) + X_OFFSET
    vy = (contour[:, 0] + r0p).astype(np.float32) + Y_OFFSET
    n  = len(vx)

    cell_id_col.extend([cell_id] * n)
    vx_col.extend(vx.tolist())
    vy_col.extend(vy.tolist())
    lid_col.extend([label] * n)

    cy, cx = prop.centroid
    cell_rows.append({
        "cell_id":    cell_id,
        "label_id":   label,
        "x_centroid": float(cx) + X_OFFSET,
        "y_centroid": float(cy) + Y_OFFSET,
        "area":       float(prop.area),
    })

print(f"  Done: {len(cell_rows)} cells")

# ── Write cell_boundaries.parquet ─────────────────────────────────────────────
df_bounds = pd.DataFrame({
    "cell_id":  pd.array(cell_id_col, dtype="object"),
    "vertex_x": pd.array(vx_col, dtype="float32"),
    "vertex_y": pd.array(vy_col, dtype="float32"),
    "label_id": pd.array(lid_col, dtype="int64"),
})
out_bounds = OUT_DIR / "cell_boundaries.parquet"
df_bounds.to_parquet(out_bounds, index=False)
print(f"Wrote {out_bounds}  ({len(df_bounds):,} rows)")

# ── Write cells.parquet ────────────────────────────────────────────────────────
df_cells = pd.DataFrame(cell_rows)
df_cells.to_parquet(OUT_DIR / "cells.parquet", index=False)
print(f"Wrote {OUT_DIR/'cells.parquet'}  ({len(df_cells)} cells)")

# ── Assign transcripts to cells (pixel lookup) ────────────────────────────────
print("Assigning transcripts to cells...")
pf     = pq.ParquetFile(TX_PATH)
# transcripts.parquet already has cell_id — we overwrite it with BIDCell assignments
src_schema = pf.schema_arrow
# Replace cell_id field type with string (BIDCell uses "bidcell-N" strings)
fields = []
for field in src_schema:
    if field.name == "cell_id":
        fields.append(pa.field("cell_id", pa.string()))
    else:
        fields.append(field)
new_schema = pa.schema(fields)

out_tx = OUT_DIR / "transcripts.parquet"
writer = pq.ParquetWriter(out_tx, new_schema)
total_assigned = 0

for batch in pf.iter_batches(batch_size=500_000):
    df = batch.to_pandas()
    col_idx = np.clip(np.round(df["x_location"].to_numpy()).astype(int), 0, W - 1)
    row_idx = np.clip(np.round(df["y_location"].to_numpy()).astype(int), 0, H - 1)
    cell_ints = mask[row_idx, col_idx]
    df["cell_id"] = np.where(
        cell_ints > 0,
        ["bidcell-" + str(c) for c in cell_ints],
        ""
    )
    total_assigned += int((cell_ints > 0).sum())
    writer.write_table(pa.Table.from_pandas(df, schema=new_schema, preserve_index=False))
    del df

writer.close()
print(f"  {total_assigned:,} transcripts assigned to cells")
print(f"Wrote {out_tx}")

print(f"\n✅ Done! Files in: {OUT_DIR}")
print(f"\nTo import into Xenium Explorer (full outs/ dataset):")
print(f"  File → Import Segmentation → {out_bounds}")
