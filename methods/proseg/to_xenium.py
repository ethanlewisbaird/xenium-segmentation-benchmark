"""
Convert proseg output → Xenium-compatible parquets and xeniumranger inputs.

Reads proseg's transcript-metadata.csv.gz (transcript_id, assignment, ...),
cell-polygons.geojson.gz (subset-space MultiPolygon features), and
cell-metadata.csv.gz (centroids), then applies subset→Xenium coordinate
offsets and writes the standard set of output files expected by downstream steps.

Proseg output columns (actual format):
  transcript-metadata:  transcript_id, assignment (float, NaN=unassigned), x, y, gene, ...
  cell-polygons:        GeoJSON FeatureCollection, MultiPolygon, properties.cell = int cell ID
  cell-metadata:        cell (int), centroid_x, centroid_y, volume, ...

Writes into <output_dir>/:
  cell_boundaries.parquet  – polygon vertices in Xenium coords
  cells.parquet            – per-cell centroids / metadata
  transcripts.parquet      – original transcripts with new cell_id column
  transcript_assignment.csv – xeniumranger --transcript-assignment input
  viz_polygons.geojson      – xeniumranger --viz-polygons input

Usage:
  python methods/proseg/to_xenium.py \
      --transcript-metadata proseg/transcript-metadata.csv.gz \
      --cell-polygons       proseg/cell-polygons.geojson.gz \
      --cell-metadata       proseg/cell-metadata.csv.gz \
      --transcripts         outs_subset/transcripts.parquet \
      --offsets             outs_subset/subset_offsets.json \
      --output-dir          proseg_output/segmentation
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def _load_geojson(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def _extract_rings(geometry: dict) -> list[np.ndarray]:
    """Return outer rings from a Polygon or MultiPolygon geometry as list of (N,2) arrays."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        # coords = [outer_ring, ...inner_rings]
        return [np.array(coords[0], dtype=np.float64)]
    if gtype == "MultiPolygon":
        # coords = [polygon, ...] where polygon = [outer_ring, ...inner_rings]
        return [np.array(poly[0], dtype=np.float64) for poly in coords if poly]
    return []


def proseg_to_xenium(
    transcript_metadata_path: Path,
    cell_polygons_path: Path,
    cell_metadata_path: Path | None,
    transcripts_path: Path,
    offsets_path: Path,
    output_dir: Path,
    min_tx: int = 5,
    max_verts: int = 300,
) -> None:
    offsets = json.loads(offsets_path.read_text())
    x_off = offsets["x_offset_um"]
    y_off = offsets["y_offset_um"]
    print(f"Coordinate offset to original space: +{x_off:.4f} µm x, +{y_off:.4f} µm y")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load transcript metadata ───────────────────────────────────────────
    print("Loading proseg transcript metadata...")
    df_meta = pd.read_csv(transcript_metadata_path)
    print(f"  Columns: {list(df_meta.columns)}")
    print(f"  Rows: {len(df_meta):,}")

    # proseg uses 'assignment' (float, NaN = background/unassigned)
    if "assignment" not in df_meta.columns:
        raise KeyError(
            f"Expected 'assignment' column in transcript metadata. "
            f"Available: {list(df_meta.columns)}"
        )

    df_assigned = df_meta[df_meta["assignment"].notna()].copy()
    df_assigned["cell_int"] = df_assigned["assignment"].astype(int)
    print(f"  Assigned: {len(df_assigned):,} / {len(df_meta):,}")

    # ── 2. Apply min-tx filter ────────────────────────────────────────────────
    counts = df_assigned.groupby("cell_int").size()
    keep_ints = counts[counts >= min_tx].index
    df_assigned = df_assigned[df_assigned["cell_int"].isin(keep_ints)]
    print(f"  Cells with >= {min_tx} transcripts: {len(keep_ints):,}")

    # Integer cell IDs → string names "proseg-{N}"
    cell_int_to_str = {c: f"proseg-{c}" for c in keep_ints}
    df_assigned["cell_id"] = df_assigned["cell_int"].map(cell_int_to_str)
    tx_to_cell = dict(zip(df_assigned["transcript_id"].astype(str), df_assigned["cell_id"]))

    # ── 3. Load cell polygons GeoJSON ─────────────────────────────────────────
    print(f"Loading cell polygons from {cell_polygons_path.name}...")
    geojson = _load_geojson(cell_polygons_path)
    features = geojson.get("features", [])
    print(f"  {len(features):,} polygon features")

    # ── 4. Load cell metadata (centroids) ─────────────────────────────────────
    cell_centroids: dict[int, tuple[float, float]] = {}
    if cell_metadata_path and cell_metadata_path.exists():
        print("Loading cell metadata...")
        df_cm = pd.read_csv(cell_metadata_path)
        for _, row in df_cm.iterrows():
            cell_centroids[int(row["cell"])] = (float(row["centroid_x"]), float(row["centroid_y"]))

    # ── 5. Build cell_boundaries.parquet from polygons ────────────────────────
    print("Building cell_boundaries.parquet...")
    cell_id_col, vx_col, vy_col, lid_col = [], [], [], []
    cell_rows = []
    label_id = 0

    for feat in tqdm(features, desc="  polygons"):
        props = feat.get("properties") or {}
        raw_id = props.get("cell")
        if raw_id is None:
            continue
        raw_id = int(raw_id)
        if raw_id not in cell_int_to_str:
            continue

        cell_str = cell_int_to_str[raw_id]
        geom = feat.get("geometry", {})
        rings = _extract_rings(geom)
        if not rings:
            continue

        # Use the largest ring (by vertex count) as the boundary
        ring_pts = max(rings, key=len)

        # Close ring: if first == last, drop the duplicate last vertex
        if len(ring_pts) > 1 and np.allclose(ring_pts[0], ring_pts[-1]):
            ring_pts = ring_pts[:-1]

        if len(ring_pts) > max_verts:
            idx = np.round(np.linspace(0, len(ring_pts) - 1, max_verts)).astype(int)
            ring_pts = ring_pts[idx]

        label_id += 1
        vx = ring_pts[:, 0].astype(np.float32) + x_off
        vy = ring_pts[:, 1].astype(np.float32) + y_off
        n = len(vx)

        cell_id_col.extend([cell_str] * n)
        vx_col.extend(vx.tolist())
        vy_col.extend(vy.tolist())
        lid_col.extend([label_id] * n)

        if raw_id in cell_centroids:
            cx = cell_centroids[raw_id][0] + x_off
            cy = cell_centroids[raw_id][1] + y_off
        else:
            cx = float(ring_pts[:, 0].mean()) + x_off
            cy = float(ring_pts[:, 1].mean()) + y_off

        cell_rows.append({
            "cell_id": cell_str,
            "label_id": label_id,
            "x_centroid": cx,
            "y_centroid": cy,
            "transcript_counts": int(counts.get(raw_id, 0)),
        })

    print(f"  Done: {len(cell_rows)} cells with boundaries")

    df_bounds = pd.DataFrame({
        "cell_id": pd.array(cell_id_col, dtype="object"),
        "vertex_x": pd.array(vx_col, dtype="float32"),
        "vertex_y": pd.array(vy_col, dtype="float32"),
        "label_id": pd.array(lid_col, dtype="int64"),
    })
    out_bounds = output_dir / "cell_boundaries.parquet"
    df_bounds.to_parquet(out_bounds, index=False)
    print(f"Wrote {out_bounds}  ({len(df_bounds):,} rows)")

    # ── 6. Write cells.parquet ────────────────────────────────────────────────
    df_cells = pd.DataFrame(cell_rows)
    df_cells.to_parquet(output_dir / "cells.parquet", index=False)
    print(f"Wrote {output_dir / 'cells.parquet'}  ({len(df_cells)} cells)")

    # ── 7. Write transcripts.parquet with cell_id ─────────────────────────────
    print("Writing transcripts.parquet (with cell_id)...")
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
        bdf = batch.to_pandas()
        bdf["cell_id"] = bdf["transcript_id"].astype(str).map(tx_to_cell).fillna("")
        total_assigned += int((bdf["cell_id"] != "").sum())
        writer.write_table(
            pa.Table.from_pandas(bdf, schema=new_schema, preserve_index=False)
        )
    writer.close()
    print(f"  {total_assigned:,} transcripts assigned to cells")
    print(f"Wrote {out_tx}")

    # ── 8. Write transcript_assignment.csv for xeniumranger ──────────────────
    print("Writing transcript_assignment.csv...")
    pf2 = pq.ParquetFile(transcripts_path)
    batches = []
    for batch in pf2.iter_batches(batch_size=500_000, columns=["transcript_id"]):
        bdf = batch.to_pandas()
        bdf["cell"] = bdf["transcript_id"].astype(str).map(tx_to_cell).fillna("")
        bdf["is_noise"] = (bdf["cell"] == "").astype(int)
        bdf["transcript_id"] = pd.to_numeric(bdf["transcript_id"], errors="coerce").astype(np.int64)
        batches.append(bdf[["transcript_id", "cell", "is_noise"]])
    df_csv = pd.concat(batches, ignore_index=True)
    out_csv = output_dir / "transcript_assignment.csv"
    df_csv.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df_csv):,} rows, {out_csv.stat().st_size / 1e6:.1f} MB)")

    # ── 9. Write viz_polygons.geojson for xeniumranger ────────────────────────
    print("Writing viz_polygons.geojson...")
    valid_strs = {r["cell_id"] for r in cell_rows}
    viz_features = []
    for feat in features:
        props = feat.get("properties") or {}
        raw_id = props.get("cell")
        if raw_id is None:
            continue
        raw_id = int(raw_id)
        cell_str = cell_int_to_str.get(raw_id)
        if cell_str not in valid_strs:
            continue
        geom = feat.get("geometry", {})
        rings = _extract_rings(geom)
        if not rings:
            continue
        ring_pts = max(rings, key=len)
        if len(ring_pts) > 1 and np.allclose(ring_pts[0], ring_pts[-1]):
            ring_pts = ring_pts[:-1]
        if len(ring_pts) > max_verts:
            idx = np.round(np.linspace(0, len(ring_pts) - 1, max_verts)).astype(int)
            ring_pts = ring_pts[idx]
        coords = [[round(float(p[0]) + x_off, 6), round(float(p[1]) + y_off, 6)] for p in ring_pts]
        coords.append(coords[0])  # close ring
        viz_features.append({
            "type": "Feature",
            "id": cell_str,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"cell_id": cell_str},
        })
    out_viz = output_dir / "viz_polygons.geojson"
    with open(out_viz, "w") as f:
        json.dump({"type": "FeatureCollection", "features": viz_features}, f)
    print(f"Wrote {out_viz}  ({len(viz_features):,} features, {out_viz.stat().st_size / 1e6:.1f} MB)")

    print(f"\nDone! Files in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--transcript-metadata", required=True, type=Path,
        help="Path to proseg transcript-metadata.csv.gz",
    )
    parser.add_argument(
        "--cell-polygons", required=True, type=Path,
        help="Path to proseg cell-polygons.geojson.gz",
    )
    parser.add_argument(
        "--cell-metadata", type=Path, default=None,
        help="Path to proseg cell-metadata.csv.gz (optional, for accurate centroids)",
    )
    parser.add_argument(
        "--transcripts", required=True, type=Path,
        help="Path to outs_subset/transcripts.parquet",
    )
    parser.add_argument(
        "--offsets", required=True, type=Path,
        help="Path to subset_offsets.json",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for output files",
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
    proseg_to_xenium(
        transcript_metadata_path=args.transcript_metadata,
        cell_polygons_path=args.cell_polygons,
        cell_metadata_path=args.cell_metadata,
        transcripts_path=args.transcripts,
        offsets_path=args.offsets,
        output_dir=args.output_dir,
        min_tx=args.min_tx,
        max_verts=args.max_verts,
    )


if __name__ == "__main__":
    main()
