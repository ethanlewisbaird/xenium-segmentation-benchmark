"""
Convert Segger output → Baysor-format CSV + GeoJSON for xeniumranger import-segmentation.

xeniumranger --transcript-assignment expects:
  transcript_id (uint64), cell (str like "seg-N" or "" unassigned), is_noise (bool)

xeniumranger --viz-polygons expects a GeoJSON FeatureCollection where each Feature
  "id" matches the "cell" string in the CSV.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

TX_PATH    = Path("segger_output/segger_dec25_0.5_False_4_12.0_5_5.0_20260403/segger_transcripts.parquet")
BOUNDS_PATH = Path("outs_subset/segger_segmentation/cell_boundaries.parquet")
OUT_CSV    = Path("segger_transcript_assignment.csv")
OUT_GJ     = Path("segger_viz_polygons.geojson")

MIN_TX = 5

# ── Load transcripts ───────────────────────────────────────────────────────────
print("Loading transcripts...")
df = pd.read_parquet(TX_PATH, columns=["transcript_id", "segger_cell_id"])
print(f"  {len(df):,} rows")

# ── Build cell ID mapping: segger_id → "seg-N" ────────────────────────────────
counts = df.groupby("segger_cell_id").size()
assigned_mask = df["segger_cell_id"] != "UNASSIGNED"
keep_cells = counts[(counts >= MIN_TX) & (counts.index != "UNASSIGNED")].index

print(f"  Cells with ≥{MIN_TX} transcripts: {len(keep_cells):,}")

# Sequential integer IDs 1..N
cell_map = {cid: f"seg-{i+1}" for i, cid in enumerate(keep_cells)}

# ── Write transcript assignment CSV ───────────────────────────────────────────
print("Building transcript assignment CSV...")
df["cell"] = df["segger_cell_id"].map(cell_map).fillna("")
df["is_noise"] = (~assigned_mask | ~df["segger_cell_id"].isin(keep_cells)).astype(int)
df["transcript_id"] = df["transcript_id"].astype(np.int64)

out = df[["transcript_id", "cell", "is_noise"]]
out.to_csv(OUT_CSV, index=False)
print(f"  Wrote {OUT_CSV}  ({len(out):,} rows, {OUT_CSV.stat().st_size/1e9:.2f} GB)")

# ── Write viz polygons GeoJSON using new IDs ──────────────────────────────────
print("Building viz polygons GeoJSON...")
bounds = pd.read_parquet(BOUNDS_PATH)

features = []
for orig_id, new_id in tqdm(cell_map.items(), total=len(cell_map)):
    grp = bounds[bounds["cell_id"] == orig_id]
    if len(grp) == 0:
        continue
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

geojson = {"type": "FeatureCollection", "features": features}
with open(OUT_GJ, "w") as f:
    json.dump(geojson, f)
print(f"  Wrote {OUT_GJ}  ({len(features):,} features, {OUT_GJ.stat().st_size/1e6:.1f} MB)")
print("Done.")
