"""
Run the full proseg pipeline: proseg → to_xenium → make_geojson.

Requires proseg to be installed:
  cargo install proseg          # via Rust (recommended)
  conda install -c bioconda proseg  # via conda

Usage:
  python methods/proseg/run_pipeline.py \
      --xenium-dir  /path/to/outs_subset \
      --output-dir  /path/to/proseg_output \
      [--offsets    /path/to/outs_subset/subset_offsets.json] \
      [--nthreads 8] [--samples 1000] [--burnin-samples 500] [--min-tx 5]

Output structure:
  <output_dir>/
  ├── proseg/
  │   ├── output.zarr                 proseg native output (SpatialData)
  │   ├── transcript-metadata.csv.gz  transcript→cell assignments (cols: transcript_id, assignment, ...)
  │   ├── cell-metadata.csv.gz        per-cell centroids and volume
  │   └── cell-polygons.geojson.gz    cell boundaries in subset coords (MultiPolygon)
  ├── segmentation/
  │   ├── cell_boundaries.parquet     Xenium-coord boundaries
  │   ├── cells.parquet               per-cell metadata
  │   ├── transcripts.parquet         transcripts with cell_id
  │   ├── transcript_assignment.csv   xeniumranger --transcript-assignment
  │   └── viz_polygons.geojson        xeniumranger --viz-polygons
  ├── cells.geojson                   reference GeoJSON for Xenium Explorer
  └── logs/
      ├── 01_proseg.log
      ├── 02_to_xenium.log
      └── 03_make_geojson.log

Next step (after pipeline completes):
  python xeniumranger/import_segmentation.py proseg \
      --id experiment_proseg \
      --xenium-bundle /path/to/outs_subset_bundle \
      --transcript-assignment <output_dir>/segmentation/transcript_assignment.csv \
      --viz-polygons <output_dir>/segmentation/viz_polygons.geojson
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run_step(cmd: list, log_path: Path, step_name: str) -> None:
    """Run a command, printing output to both the terminal and a log file."""
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"  Log → {log_path}")
    print(f"{'='*60}")
    print("  $", " ".join(str(c) for c in cmd), "\n")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"\n[ERROR] {step_name} failed (exit code {proc.returncode})")
        print(f"        See log: {log_path}")
        sys.exit(proc.returncode)


def find_proseg() -> str:
    found = shutil.which("proseg")
    if found:
        return found
    raise FileNotFoundError(
        "proseg not found on PATH. Install via:\n"
        "  cargo install proseg\n"
        "  conda install -c bioconda proseg\n"
        "Then ensure the binary is on PATH."
    )


def run_pipeline(
    xenium_dir: Path,
    output_dir: Path,
    offsets: Path,
    nthreads: int,
    samples: int,
    burnin_samples: int,
    min_tx: int,
    max_verts: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    proseg_dir = output_dir / "proseg"
    seg_dir = output_dir / "segmentation"
    proseg_dir.mkdir(parents=True, exist_ok=True)

    proseg_bin = find_proseg()
    py = sys.executable

    transcripts = xenium_dir / "transcripts.parquet"
    if not transcripts.exists():
        transcripts = xenium_dir / "transcripts.csv.gz"
    if not transcripts.exists():
        print(f"[ERROR] No transcripts.parquet or transcripts.csv.gz in {xenium_dir}")
        sys.exit(1)

    zarr_out = proseg_dir / "output.zarr"
    tx_meta = proseg_dir / "transcript-metadata.csv.gz"
    cell_meta = proseg_dir / "cell-metadata.csv.gz"
    polygons = proseg_dir / "cell-polygons.geojson.gz"

    # proseg requires recorded-samples <= samples; default recorded-samples=100
    recorded = min(100, samples)

    # ── Step 1: Run proseg ────────────────────────────────────────────────────
    proseg_cmd = [
        proseg_bin, "--xenium", str(transcripts),
        "--nthreads",                str(nthreads),
        "--samples",                 str(samples),
        "--burnin-samples",          str(burnin_samples),
        "--recorded-samples",        str(recorded),
        "--output-spatialdata",      str(zarr_out),
        "--output-transcript-metadata", str(tx_meta),
        "--output-cell-polygons",    str(polygons),
        "--output-cell-metadata",    str(cell_meta),
        "--overwrite",
    ]
    run_step(
        proseg_cmd,
        logs_dir / "01_proseg.log",
        "Step 1/3 — Run proseg",
    )

    # ── Step 2: Convert outputs to Xenium coordinates ─────────────────────────
    run_step(
        [
            py, HERE / "to_xenium.py",
            "--transcript-metadata", tx_meta,
            "--cell-polygons",       polygons,
            "--cell-metadata",       cell_meta,
            "--transcripts",         xenium_dir / "transcripts.parquet",
            "--offsets",             offsets,
            "--output-dir",          seg_dir,
            "--min-tx",              str(min_tx),
            "--max-verts",           str(max_verts),
        ],
        logs_dir / "02_to_xenium.log",
        "Step 2/3 — Convert to Xenium coordinates",
    )

    # ── Step 3: Build reference GeoJSON ───────────────────────────────────────
    geojson_path = output_dir / "cells.geojson"
    run_step(
        [
            py, REPO / "converters" / "make_geojson.py",
            "--boundaries", seg_dir / "cell_boundaries.parquet",
            "--output",     geojson_path,
        ],
        logs_dir / "03_make_geojson.log",
        "Step 3/3 — Build reference GeoJSON",
    )

    print(f"\n{'='*60}")
    print("  proseg pipeline complete!")
    print(f"{'='*60}")
    print(f"  Proseg output:        {proseg_dir}/")
    print(f"  Segmentation:         {seg_dir}/")
    print(f"  Reference GeoJSON:    {geojson_path}")
    print(f"  Logs:                 {logs_dir}/")
    print()
    print("  Next step — import into Xenium Explorer:")
    print(f"    python xeniumranger/import_segmentation.py proseg \\")
    print(f"        --id experiment_proseg \\")
    print(f"        --xenium-bundle /path/to/outs_subset_bundle \\")
    print(f"        --transcript-assignment {seg_dir / 'transcript_assignment.csv'} \\")
    print(f"        --viz-polygons {seg_dir / 'viz_polygons.geojson'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--xenium-dir", required=True, type=Path,
                        help="Path to outs_subset/ directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write all proseg outputs")
    parser.add_argument("--offsets", type=Path, default=None,
                        help="Path to subset_offsets.json "
                             "(default: <xenium-dir>/subset_offsets.json)")
    parser.add_argument("--nthreads", type=int, default=8,
                        help="Threads for proseg (default: 8)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Proseg MCMC samples (default: 1000)")
    parser.add_argument("--burnin-samples", type=int, default=500,
                        help="Proseg MCMC burn-in samples (default: 500)")
    parser.add_argument("--min-tx", type=int, default=5,
                        help="Minimum transcripts per cell (default: 5)")
    parser.add_argument("--max-verts", type=int, default=300,
                        help="Maximum polygon vertices per cell (default: 300)")
    args = parser.parse_args()

    offsets = args.offsets or (args.xenium_dir / "subset_offsets.json")
    if not offsets.exists():
        parser.error(f"subset_offsets.json not found at {offsets}. "
                     "Pass --offsets or ensure xenium-dir contains subset_offsets.json.")

    run_pipeline(
        xenium_dir=args.xenium_dir,
        output_dir=args.output_dir,
        offsets=offsets,
        nthreads=args.nthreads,
        samples=args.samples,
        burnin_samples=args.burnin_samples,
        min_tx=args.min_tx,
        max_verts=args.max_verts,
    )


if __name__ == "__main__":
    main()
