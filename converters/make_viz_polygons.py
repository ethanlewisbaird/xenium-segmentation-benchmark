"""
Build viz_polygons GeoJSON from Segger boundaries for xeniumranger --viz-polygons.

Reads the Segger transcript parquet to build the seg-N cell ID mapping, then
writes a GeoJSON where each feature's "id" matches the "cell" column in the
Baysor-format CSV (produced by make_baysor_format.py).

Usage:
  python make_viz_polygons.py \
      --boundaries     outs_subset/segger_segmentation/cell_boundaries.parquet \
      --segger-parquet segger_output/.../segger_transcripts.parquet \
      --output         segger_viz_polygons.geojson \
      [--min-tx 5]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def make_viz_polygons(
    boundaries_path: Path,
    parquet_path: Path,
    output_path: Path,
    min_tx: int = 5,
) -> None:
    print("Building cell ID map from transcript parquet...")
    counts = (
        pd.read_parquet(parquet_path, columns=["segger_cell_id"])
        .groupby("segger_cell_id")
        .size()
    )
    keep_cells = counts[(counts >= min_tx) & (counts.index != "UNASSIGNED")].index
    cell_map = {cid: f"seg-{i + 1}" for i, cid in enumerate(keep_cells)}
    print(f"  {len(cell_map):,} cells")

    print("Loading boundaries...")
    bounds = pd.read_parquet(boundaries_path)
    bounds = bounds[bounds["cell_id"].isin(cell_map)]

    features = []
    for orig_id, grp in tqdm(
        bounds.groupby("cell_id", sort=False), total=len(cell_map)
    ):
        new_id = cell_map[orig_id]
        vx = grp["vertex_x"].values.tolist()
        vy = grp["vertex_y"].values.tolist()
        coords = [[float(x), float(y)] for x, y in zip(vx, vy)]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "id": new_id,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"cell_id": new_id},
        })

    print(f"Writing {output_path} ({len(features):,} features)...")
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"Done: {output_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--boundaries", required=True,
        help="Path to cell_boundaries.parquet (original Xenium coordinates)",
    )
    parser.add_argument(
        "--segger-parquet", required=True,
        help="Path to segger_transcripts.parquet (to build cell ID mapping)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output GeoJSON path for xeniumranger --viz-polygons",
    )
    parser.add_argument(
        "--min-tx", type=int, default=5,
        help="Minimum transcripts per cell to include (default: 5)",
    )
    args = parser.parse_args()
    make_viz_polygons(
        Path(args.boundaries),
        Path(args.segger_parquet),
        Path(args.output),
        args.min_tx,
    )


if __name__ == "__main__":
    main()
