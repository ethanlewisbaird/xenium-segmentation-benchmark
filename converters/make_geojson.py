"""
Convert cell_boundaries.parquet → GeoJSON FeatureCollection for xeniumranger.

Coordinates in the parquet must already be in original Xenium µm space
(i.e. the to_xenium.py offset has been applied).

Usage:
  python make_geojson.py \
      --boundaries outs_subset/bidcell_segmentation/cell_boundaries.parquet \
      --output     bidcell_cells.geojson
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def make_geojson(boundaries_path: Path, output_path: Path) -> None:
    print(f"Reading {boundaries_path}...")
    df = pd.read_parquet(boundaries_path)
    print(f"  {len(df):,} rows, {df['cell_id'].nunique():,} cells")

    features = []
    for cell_id, grp in tqdm(df.groupby("cell_id", sort=False), desc="Building GeoJSON"):
        vx = grp["vertex_x"].values.tolist()
        vy = grp["vertex_y"].values.tolist()
        coords = [[float(x), float(y)] for x, y in zip(vx, vy)]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "id": str(cell_id),
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"cell_id": str(cell_id)},
        })

    geojson = {"type": "FeatureCollection", "features": features}

    print(f"Writing {output_path} ({len(features):,} features)...")
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"Done: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--boundaries", required=True,
        help="Path to cell_boundaries.parquet (original Xenium coordinates)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output GeoJSON path (e.g. bidcell_cells.geojson)",
    )
    args = parser.parse_args()
    make_geojson(Path(args.boundaries), Path(args.output))


if __name__ == "__main__":
    main()
