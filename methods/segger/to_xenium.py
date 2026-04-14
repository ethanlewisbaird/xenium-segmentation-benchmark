"""
Convert Segger transcript parquet → Xenium-compatible cell_boundaries.parquet.

Reads segger_transcripts.parquet, computes per-cell convex hull boundaries,
and writes cell_boundaries.parquet + cells.parquet + transcripts.parquet to
the output directory.  Coordinates are shifted from subset space to original
Xenium space using offsets from subset_offsets.json.

Usage:
  python methods/segger/to_xenium.py \
      --segger-parquet  segger_output/.../segger_transcripts.parquet \
      --offsets         outs_subset/subset_offsets.json \
      --output-dir      outs_subset/segger_segmentation \
      [--min-tx 5] [--max-verts 300]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import ConvexHull
from tqdm import tqdm


def segger_to_xenium(
    parquet_path: Path,
    offsets_path: Path,
    output_dir: Path,
    min_tx: int = 5,
    max_verts: int = 300,
) -> None:
    offsets = json.loads(offsets_path.read_text())
    x_offset = offsets["x_offset_um"]
    y_offset = offsets["y_offset_um"]
    print(f"Coordinate offset to original space: +{x_offset:.2f} µm x, +{y_offset:.2f} µm y")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading segger transcripts...")
    df = pd.read_parquet(
        parquet_path,
        columns=["transcript_id", "x_location", "y_location",
                 "feature_name", "segger_cell_id", "bound"],
    )

    df = df[df["segger_cell_id"] != "UNASSIGNED"].copy()
    print(f"  Assigned transcripts: {len(df):,}")

    counts = df.groupby("segger_cell_id").size()
    keep = counts[counts >= min_tx].index
    df = df[df["segger_cell_id"].isin(keep)]
    print(f"  Cells with >= {min_tx} transcripts: {len(keep):,}")
    print(f"  Transcripts in kept cells: {len(df):,}")

    print("Computing convex hull boundaries per cell...")
    cell_id_col, vx_col, vy_col, lid_col = [], [], [], []
    cell_rows = []
    label_id = 0

    for cell_id, grp in tqdm(df.groupby("segger_cell_id", sort=False), total=len(keep)):
        label_id += 1
        pts = grp[["x_location", "y_location"]].values.astype(np.float64)
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()

        if len(pts) < 3:
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            vx = (cx + 3.0 * np.cos(angles)).astype(np.float32) + x_offset
            vy = (cy + 3.0 * np.sin(angles)).astype(np.float32) + y_offset
        else:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
            except Exception:
                x0, x1 = pts[:, 0].min(), pts[:, 0].max()
                y0, y1 = pts[:, 1].min(), pts[:, 1].max()
                hull_pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

            if len(hull_pts) > max_verts:
                idx = np.round(np.linspace(0, len(hull_pts) - 1, max_verts)).astype(int)
                hull_pts = hull_pts[idx]

            vx = hull_pts[:, 0].astype(np.float32) + x_offset
            vy = hull_pts[:, 1].astype(np.float32) + y_offset

        n = len(vx)
        cell_id_col.extend([cell_id] * n)
        vx_col.extend(vx.tolist())
        vy_col.extend(vy.tolist())
        lid_col.extend([label_id] * n)
        cell_rows.append({
            "cell_id": cell_id,
            "label_id": label_id,
            "x_centroid": float(cx) + x_offset,
            "y_centroid": float(cy) + y_offset,
            "transcript_counts": len(grp),
        })

    print(f"  Done: {label_id} cells")

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

    # Write transcripts.parquet with segger_cell_id as cell_id
    print("Writing transcripts.parquet...")
    pf = pq.ParquetFile(parquet_path)
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
        bdf = batch.to_pandas()
        bdf["cell_id"] = bdf["segger_cell_id"].where(
            bdf["segger_cell_id"] != "UNASSIGNED", ""
        )
        total_assigned += int((bdf["cell_id"] != "").sum())
        writer.write_table(
            pa.Table.from_pandas(bdf, schema=new_schema, preserve_index=False)
        )
    writer.close()
    print(f"  {total_assigned:,} transcripts assigned to cells")
    print(f"Wrote {out_tx}")
    print(f"\nDone! Files in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--segger-parquet", required=True,
        help="Path to segger_transcripts.parquet",
    )
    parser.add_argument(
        "--offsets", required=True,
        help="Path to subset_offsets.json (from xenium-subsetter)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write cell_boundaries.parquet, cells.parquet, transcripts.parquet",
    )
    parser.add_argument(
        "--min-tx", type=int, default=5,
        help="Minimum transcripts per cell (default: 5)",
    )
    parser.add_argument(
        "--max-verts", type=int, default=300,
        help="Maximum polygon vertices per cell (default: 300)",
    )
    args = parser.parse_args()
    segger_to_xenium(
        Path(args.segger_parquet),
        Path(args.offsets),
        Path(args.output_dir),
        args.min_tx,
        args.max_verts,
    )


if __name__ == "__main__":
    main()
