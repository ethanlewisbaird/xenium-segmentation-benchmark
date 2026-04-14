"""
Fast: Build viz_polygons GeoJSON from cell_boundaries.parquet using new "seg-N" IDs.
The cell_map (segger_id → "seg-N") was already computed; we just need to apply it.
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BOUNDS_PATH  = Path("outs_subset/segger_segmentation/cell_boundaries.parquet")
TX_PATH      = Path("segger_output/segger_dec25_0.5_False_4_12.0_5_5.0_20260403/segger_transcripts.parquet")
OUT_GJ       = Path("segger_viz_polygons.geojson")
MIN_TX = 5

# Rebuild cell_map from transcript counts (fast)
print("Building cell_map...")
counts = (pd.read_parquet(TX_PATH, columns=["segger_cell_id"])
            .groupby("segger_cell_id").size())
keep_cells = counts[(counts >= MIN_TX) & (counts.index != "UNASSIGNED")].index
cell_map = {cid: f"seg-{i+1}" for i, cid in enumerate(keep_cells)}
print(f"  {len(cell_map):,} cells")

# Load boundaries and build GeoJSON using groupby (fast path)
print("Loading boundaries...")
bounds = pd.read_parquet(BOUNDS_PATH)
bounds = bounds[bounds["cell_id"].isin(cell_map)]

features = []
for orig_id, grp in tqdm(bounds.groupby("cell_id", sort=False), total=len(cell_map)):
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
        "properties": {"cell_id": new_id}
    })

print(f"Writing {OUT_GJ} ({len(features):,} features)...")
with open(OUT_GJ, "w") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f)
print(f"Done: {OUT_GJ.stat().st_size/1e6:.1f} MB")
