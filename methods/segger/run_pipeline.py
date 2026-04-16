"""
Run the full Segger pipeline: create_dataset → train → predict → to_xenium → converters.

All intermediate outputs are placed under a single output directory with a
predictable subdirectory layout.  Each step's stdout/stderr is written to a
numbered log file under <output_dir>/logs/ as well as printed to the terminal.

Usage:
  conda run -n segger311 python methods/segger/run_pipeline.py \\
      --xenium-dir  /path/to/outs_subset \\
      --output-dir  /path/to/segger_output \\
      [--offsets    /path/to/outs_subset/subset_offsets.json] \\
      [--epochs 100] [--batch-size 1] [--devices 1]

Output structure:
  <output_dir>/
  ├── tiles/                 graph tiles (create_dataset)
  ├── model/                 Lightning checkpoints (train)
  ├── predict/               Segger output parquet (run_predict)
  ├── segmentation/          cell_boundaries.parquet, cells.parquet (to_xenium)
  ├── transcript_assignment.csv
  ├── viz_polygons.geojson
  └── logs/
      ├── 01_create_dataset.log
      ├── 02_train.log
      ├── 03_run_predict.log
      ├── 04_to_xenium.log
      └── 05_converters.log
"""

from __future__ import annotations

import argparse
import glob
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


def find_transcripts_df(predict_dir: Path) -> Path:
    """Find transcripts_df.parquet inside the nested predict output directory."""
    matches = list(predict_dir.glob("**/transcripts_df.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"transcripts_df.parquet not found anywhere under {predict_dir}"
        )
    if len(matches) > 1:
        print(f"  WARNING: multiple transcripts_df.parquet found; using {matches[-1]}")
    return matches[-1]


def find_latest_checkpoint_version(model_dir: Path) -> Path:
    """Return the latest lightning_logs/version_N directory."""
    versions = sorted(
        (model_dir / "lightning_logs").glob("version_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not versions:
        raise FileNotFoundError(f"No lightning_logs/version_* found in {model_dir}")
    return versions[-1]


def run_pipeline(
    xenium_dir: Path,
    output_dir: Path,
    offsets: Path,
    epochs: int,
    batch_size: int,
    devices: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

    tiles_dir = output_dir / "tiles"
    model_dir = output_dir / "model"
    predict_dir = output_dir / "predict"
    seg_dir = output_dir / "segmentation"

    py = sys.executable

    # ── Step 1: Build graph tiles ─────────────────────────────────────────────
    run_step(
        [py, HERE / "create_dataset.py",
         "--xenium-dir", xenium_dir,
         "--output-dir", tiles_dir],
        logs_dir / "01_create_dataset.log",
        "Step 1/5 — Create dataset (graph tiles)",
    )

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    run_step(
        [py, HERE / "train.py",
         "--tiles-dir",  tiles_dir,
         "--output-dir", model_dir,
         "--epochs",     epochs,
         "--batch-size", batch_size,
         "--devices",    devices],
        logs_dir / "02_train.log",
        "Step 2/5 — Train GNN",
    )

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    checkpoint_version = find_latest_checkpoint_version(model_dir)
    run_step(
        [py, HERE / "run_predict.py",
         "--dataset-dir", tiles_dir,
         "--output-dir",  predict_dir,
         "--checkpoint",  checkpoint_version,
         "--transcripts", xenium_dir / "transcripts.parquet"],
        logs_dir / "03_run_predict.log",
        "Step 3/5 — Run inference",
    )

    # ── Step 4: Convert to Xenium coordinates ─────────────────────────────────
    transcripts_df = find_transcripts_df(predict_dir)
    print(f"\n  Found Segger output: {transcripts_df}")
    run_step(
        [py, HERE / "to_xenium.py",
         "--segger-parquet", transcripts_df,
         "--transcripts",    xenium_dir / "transcripts.parquet",
         "--offsets",        offsets,
         "--output-dir",     seg_dir],
        logs_dir / "04_to_xenium.log",
        "Step 4/5 — Convert to Xenium coordinates",
    )

    # ── Step 5a: Baysor-format transcript assignment CSV ──────────────────────
    transcript_csv = output_dir / "transcript_assignment.csv"
    run_step(
        [py, REPO / "converters" / "make_baysor_format.py",
         "--segger-parquet", transcripts_df,
         "--output-csv",     transcript_csv],
        logs_dir / "05a_make_baysor_format.log",
        "Step 5a/5 — Make transcript assignment CSV",
    )

    # ── Step 5b: Viz polygons GeoJSON ─────────────────────────────────────────
    viz_geojson = output_dir / "viz_polygons.geojson"
    run_step(
        [py, REPO / "converters" / "make_viz_polygons.py",
         "--boundaries",     seg_dir / "cell_boundaries.parquet",
         "--segger-parquet", transcripts_df,
         "--output",         viz_geojson],
        logs_dir / "05b_make_viz_polygons.log",
        "Step 5b/5 — Make viz polygons GeoJSON",
    )

    print(f"\n{'='*60}")
    print("  Segger pipeline complete!")
    print(f"{'='*60}")
    print(f"  Segmentation:          {seg_dir}/")
    print(f"  Transcript assignment: {transcript_csv}")
    print(f"  Viz polygons:          {viz_geojson}")
    print(f"  Logs:                  {logs_dir}/")
    print()
    print("  Next step — import into Xenium Explorer:")
    print(f"    python xeniumranger/import_segmentation.py segger \\")
    print(f"        --id experiment_segger \\")
    print(f"        --xenium-bundle /path/to/outs_subset_bundle \\")
    print(f"        --transcript-assignment {transcript_csv} \\")
    print(f"        --viz-polygons {viz_geojson}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--xenium-dir", required=True, type=Path,
                        help="Path to outs_subset/ directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write all Segger outputs")
    parser.add_argument("--offsets", type=Path, default=None,
                        help="Path to subset_offsets.json "
                             "(default: <xenium-dir>/subset_offsets.json)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size (default: 1)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    args = parser.parse_args()

    offsets = args.offsets or (args.xenium_dir / "subset_offsets.json")
    if not offsets.exists():
        parser.error(f"subset_offsets.json not found at {offsets}. "
                     "Pass --offsets or ensure xenium-dir contains subset_offsets.json.")

    run_pipeline(
        xenium_dir=args.xenium_dir,
        output_dir=args.output_dir,
        offsets=offsets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
