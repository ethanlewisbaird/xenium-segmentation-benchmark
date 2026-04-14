"""
Convert BIDCell segmentation mask → Xenium-compatible cell_boundaries.parquet.

Reads the BIDCell output mask (uint32 TIF), extracts per-cell contour polygons,
and writes cell_boundaries.parquet + cells.parquet + transcripts.parquet to
the output directory.  Coordinates are shifted from subset space to original
Xenium space using offsets from subset_offsets.json.

Usage:
  python methods/bidcell/to_xenium.py \
      --mask        outs_subset/model_outputs/.../epoch_N_step_M.tif \
      --transcripts outs_subset/transcripts.parquet \
      --offsets     outs_subset/subset_offsets.json \
      --output-dir  outs_subset/bidcell_segmentation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
from skimage.measure import find_contours, regionprops


def bidcell_to_xenium(
    mask_path: Path,
    transcripts_path: Path,
    offsets_path: Path,
    output_dir: Path,
) -> None:
    offsets = json.loads(offsets_path.read_text())
    x_offset = offsets["x_offset_um"]
    y_offset = offsets["y_offset_um"]
    print(f"Coordinate offset to original space: +{x_offset:.2f} µm x, +{y_offset:.2f} µm y")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BIDCell segmentation mask...")
    mask = tifffile.imread(mask_path)  # (H, W) uint32
    H, W = mask.shape
    print(f"  Shape: {mask.shape}, cells: {mask.max()}")

    print("Extracting cell boundaries...")
    props = regionprops(mask)

    cell_id_col, vx_col, vy_col, lid_col = [], [], [], []
    cell_rows = []

    for prop in props:
        label = prop.label
        cell_id = f"bidcell-{label}"

        r0, c0, r1, c1 = prop.bbox
        r0p = max(0, r0 - 1)
        c0p = max(0, c0 - 1)
        r1p = min(H, r1 + 1)
        c1p = min(W, c1 + 1)

        sub = (mask[r0p:r1p, c0p:c1p] == label).astype(np.uint8)
        contours = find_contours(sub, level=0.5)
        if not contours:
            continue

        contour = contours[np.argmax([len(c) for c in contours])]
        if len(contour) > 300:
            idx = np.round(np.linspace(0, len(contour) - 1, 300)).astype(int)
            contour = contour[idx]

        vx = (contour[:, 1] + c0p).astype(np.float32) + x_offset
        vy = (contour[:, 0] + r0p).astype(np.float32) + y_offset
        n = len(vx)

        cell_id_col.extend([cell_id] * n)
        vx_col.extend(vx.tolist())
        vy_col.extend(vy.tolist())
        lid_col.extend([label] * n)

        cy, cx = prop.centroid
        cell_rows.append({
            "cell_id": cell_id,
            "label_id": label,
            "x_centroid": float(cx) + x_offset,
            "y_centroid": float(cy) + y_offset,
            "area": float(prop.area),
        })

    print(f"  Done: {len(cell_rows)} cells")

    # Write cell_boundaries.parquet
    df_bounds = pd.DataFrame({
        "cell_id": pd.array(cell_id_col, dtype="object"),
        "vertex_x": pd.array(vx_col, dtype="float32"),
        "vertex_y": pd.array(vy_col, dtype="float32"),
        "label_id": pd.array(lid_col, dtype="int64"),
    })
    out_bounds = output_dir / "cell_boundaries.parquet"
    df_bounds.to_parquet(out_bounds, index=False)
    print(f"Wrote {out_bounds}  ({len(df_bounds):,} rows)")

    # Write cells.parquet
    df_cells = pd.DataFrame(cell_rows)
    df_cells.to_parquet(output_dir / "cells.parquet", index=False)
    print(f"Wrote {output_dir / 'cells.parquet'}  ({len(df_cells)} cells)")

    # Assign transcripts to cells via pixel lookup
    print("Assigning transcripts to cells...")
    pf = pq.ParquetFile(transcripts_path)
    src_schema = pf.schema_arrow
    fields = [
        pa.field("cell_id", pa.string()) if f.name == "cell_id" else f
        for f in src_schema
    ]
    new_schema = pa.schema(fields)

    out_tx = output_dir / "transcripts.parquet"
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
            "",
        )
        total_assigned += int((cell_ints > 0).sum())
        writer.write_table(
            pa.Table.from_pandas(df, schema=new_schema, preserve_index=False)
        )
    writer.close()
    print(f"  {total_assigned:,} transcripts assigned to cells")
    print(f"Wrote {out_tx}")
    print(f"\nDone! Files in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--mask", required=True,
        help="Path to BIDCell segmentation mask TIF (uint32)",
    )
    parser.add_argument(
        "--transcripts", required=True,
        help="Path to outs_subset/transcripts.parquet",
    )
    parser.add_argument(
        "--offsets", required=True,
        help="Path to subset_offsets.json (from xenium-subsetter)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write cell_boundaries.parquet, cells.parquet, transcripts.parquet",
    )
    args = parser.parse_args()
    bidcell_to_xenium(
        Path(args.mask),
        Path(args.transcripts),
        Path(args.offsets),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
