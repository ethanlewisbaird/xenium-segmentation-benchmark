"""
Convert Segger segmentation output → Xenium Explorer-compatible parquet files.
Outputs written to outs_subset/segger_segmentation/.

Usage:
  conda run -n segger311 python segger_to_xenium.py
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import glob, os
from pathlib import Path
from scipy.spatial import ConvexHull
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
TX_PATH  = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/segger_output"
                "/segger_dec25_0.5_False_4_12.0_5_5.0_20260403/segger_transcripts.parquet")
OUT_DIR  = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/outs_subset/segger_segmentation")
OUT_DIR.mkdir(exist_ok=True)

MIN_TX = 5       # minimum transcripts to keep a cell
MAX_VERTS = 300  # max polygon vertices

# ── Coordinate offset: subset → original Xenium space ─────────────────────────
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

# ── Load and filter transcripts ────────────────────────────────────────────────
print("Loading segger transcripts...")
df = pd.read_parquet(TX_PATH, columns=["transcript_id", "x_location", "y_location",
                                        "feature_name", "segger_cell_id", "bound"])

# Keep only assigned transcripts
df = df[df["segger_cell_id"] != "UNASSIGNED"].copy()
print(f"  Assigned transcripts: {len(df):,}")

# Filter cells with too few transcripts
counts = df.groupby("segger_cell_id").size()
keep   = counts[counts >= MIN_TX].index
df     = df[df["segger_cell_id"].isin(keep)]
print(f"  Cells with ≥{MIN_TX} transcripts: {len(keep):,}")
print(f"  Transcripts in kept cells: {len(df):,}")

# ── Compute convex hull boundaries per cell ────────────────────────────────────
print("Computing convex hull boundaries per cell...")

cell_id_col, vx_col, vy_col, lid_col = [], [], [], []
cell_rows = []

grouped = df.groupby("segger_cell_id", sort=False)
label_id = 0

for cell_id, grp in tqdm(grouped, total=len(keep)):
    label_id += 1
    pts = grp[["x_location", "y_location"]].values.astype(np.float64)
    cx  = pts[:, 0].mean()
    cy  = pts[:, 1].mean()

    if len(pts) < 3:
        # Just use a small circle around centroid
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        r = 3.0
        vx = (cx + r * np.cos(angles)).astype(np.float32) + X_OFFSET
        vy = (cy + r * np.sin(angles)).astype(np.float32) + Y_OFFSET
    else:
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        except Exception:
            # Degenerate (collinear); use bounding box corners
            x0, x1 = pts[:, 0].min(), pts[:, 0].max()
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            hull_pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        if len(hull_pts) > MAX_VERTS:
            idx = np.round(np.linspace(0, len(hull_pts) - 1, MAX_VERTS)).astype(int)
            hull_pts = hull_pts[idx]

        vx = hull_pts[:, 0].astype(np.float32) + X_OFFSET
        vy = hull_pts[:, 1].astype(np.float32) + Y_OFFSET

    n = len(vx)
    cell_id_col.extend([cell_id] * n)
    vx_col.extend(vx.tolist())
    vy_col.extend(vy.tolist())
    lid_col.extend([label_id] * n)

    cell_rows.append({
        "cell_id":            cell_id,
        "label_id":           label_id,
        "x_centroid":         float(cx) + X_OFFSET,
        "y_centroid":         float(cy) + Y_OFFSET,
        "transcript_counts":  len(grp),
    })

print(f"  Done: {label_id} cells")

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

# ── Write transcripts.parquet with segger_cell_id as cell_id ──────────────────
print("Writing transcripts.parquet...")
pf         = pq.ParquetFile(TX_PATH)
src_schema = pf.schema_arrow

# Replace cell_id field type with string (Segger uses random-string IDs)
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
    bdf = batch.to_pandas()
    # Use segger_cell_id as the cell_id; blank string for unassigned
    bdf["cell_id"] = bdf["segger_cell_id"].where(
        bdf["segger_cell_id"] != "UNASSIGNED", ""
    )
    total_assigned += int((bdf["cell_id"] != "").sum())
    writer.write_table(pa.Table.from_pandas(bdf, schema=new_schema, preserve_index=False))
    del bdf

writer.close()
print(f"  {total_assigned:,} transcripts assigned to cells")
print(f"Wrote {out_tx}")

print(f"\nDone! Files in: {OUT_DIR}")
print(f"\nTo import into Xenium Explorer (full outs/ dataset):")
print(f"  File → Import Segmentation → {out_bounds}")
