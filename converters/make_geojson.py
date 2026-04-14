"""
Convert cell_boundaries.parquet → GeoJSON FeatureCollection for xeniumranger.
Coordinates in output are already in original Xenium µm space.

Usage:
  python make_geojson.py bidcell   → bidcell_cells.geojson
  python make_geojson.py segger    → segger_cells.geojson
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

mode = sys.argv[1] if len(sys.argv) > 1 else "bidcell"

if mode == "bidcell":
    parquet = Path("outs_subset/bidcell_segmentation/cell_boundaries.parquet")
    out     = Path("bidcell_cells.geojson")
elif mode == "segger":
    parquet = Path("outs_subset/segger_segmentation/cell_boundaries.parquet")
    out     = Path("segger_cells.geojson")
else:
    raise ValueError(f"Unknown mode: {mode}")

print(f"Reading {parquet}...")
df = pd.read_parquet(parquet)
print(f"  {len(df):,} rows, {df['cell_id'].nunique():,} cells")

features = []
for cell_id, grp in tqdm(df.groupby("cell_id", sort=False), desc="Building GeoJSON"):
    vx = grp["vertex_x"].values.tolist()
    vy = grp["vertex_y"].values.tolist()
    # GeoJSON polygon ring must be closed (last == first)
    coords = [[float(x), float(y)] for x, y in zip(vx, vy)]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    features.append({
        "type": "Feature",
        "id": str(cell_id),
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"cell_id": str(cell_id)}
    })

geojson = {"type": "FeatureCollection", "features": features}

print(f"Writing {out} ({len(features):,} features)...")
with open(out, "w") as f:
    json.dump(geojson, f)

print(f"Done: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
