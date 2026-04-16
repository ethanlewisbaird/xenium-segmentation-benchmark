"""
Run the full BIDCell pipeline: preprocess → train → predict → to_xenium → make_geojson.

All intermediate outputs are placed under a single output directory.
A BIDCell config YAML is auto-generated from the supplied inputs.
Each step's stdout/stderr is written to a numbered log file under
<output_dir>/logs/ as well as printed to the terminal.

Usage:
  conda run -n bidcell python methods/bidcell/run_pipeline.py \\
      --xenium-dir  /path/to/outs_subset \\
      --output-dir  /path/to/bidcell_output \\
      [--offsets    /path/to/outs_subset/subset_offsets.json] \\
      [--cpus 4] [--total-steps 4000] [--total-epochs 1]

Output structure:
  <output_dir>/
  ├── config.yaml            auto-generated BIDCell config
  ├── bidcell/               BIDCell working directory (model_outputs/ inside)
  ├── segmentation/          cell_boundaries.parquet, cells.parquet (to_xenium)
  ├── cells.geojson
  └── logs/
      ├── 01_run_bidcell.log
      ├── 02_to_xenium.log
      └── 03_make_geojson.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

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


def find_latest_mask(bidcell_dir: Path) -> Path:
    """Find the most recent BIDCell prediction mask TIF."""
    masks = sorted(
        bidcell_dir.glob("model_outputs/**/*.tif"),
        key=lambda p: p.stat().st_mtime,
    )
    if not masks:
        raise FileNotFoundError(
            f"No .tif mask found under {bidcell_dir / 'model_outputs'}. "
            "BIDCell prediction may not have completed."
        )
    return masks[-1]


def write_config(
    output_dir: Path,
    xenium_dir: Path,
    bidcell_dir: Path,
    cpus: int,
    total_steps: int,
    total_epochs: int,
    fp_ref: str | None,
    fp_pos_markers: str | None,
    fp_neg_markers: str | None,
) -> Path:
    """Write a BIDCell config YAML into output_dir/config.yaml."""
    config = {
        "cpus": cpus,
        "files": {
            "data_dir": str(bidcell_dir),
            "fp_dapi": str(xenium_dir / "morphology_focus" / "ch0000_dapi.ome.tif"),
            "fp_transcripts": str(xenium_dir / "transcripts.csv.gz"),
            "fp_ref":         fp_ref or None,
            "fp_pos_markers": fp_pos_markers or None,
            "fp_neg_markers": fp_neg_markers or None,
        },
        "nuclei_fovs": {
            "stitch_nuclei_fovs": False,
        },
        "nuclei": {
            "diameter": None,
        },
        "transcripts": {
            "shift_to_origin": False,
            "x_col": "x_location",
            "y_col": "y_location",
            "gene_col": "feature_name",
            "transcripts_to_filter": [
                "NegControlProbe_",
                "antisense_",
                "NegControlCodeword_",
                "BLANK_",
                "Blank-",
                "NegPrb",
            ],
        },
        "affine": {
            "target_pix_um": 1.0,
            "base_pix_x": 0.2125,
            "base_pix_y": 0.2125,
            "base_ts_x": 1.0,
            "base_ts_y": 1.0,
            "global_shift_x": 0,
            "global_shift_y": 0,
        },
        "model_params": {
            "name": "custom",
            "patch_size": 48,
            "elongated": [],
        },
        "training_params": {
            "total_epochs": total_epochs,
            "total_steps": total_steps,
            "ne_weight": 1.0,
            "os_weight": 1.0,
            "cc_weight": 0.0 if not fp_ref else 1.0,
            "ov_weight": 1.0,
            "pos_weight": 0.0 if not fp_pos_markers else 1.0,
            "neg_weight": 0.0 if not fp_neg_markers else 1.0,
        },
        "testing_params": {
            "test_epoch": total_epochs,
            "test_step": total_steps,
        },
        "experiment_dirs": {
            "dir_id": "last",
        },
    }

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Written config: {config_path}")
    return config_path


def run_pipeline(
    xenium_dir: Path,
    output_dir: Path,
    offsets: Path,
    cpus: int,
    total_steps: int,
    total_epochs: int,
    fp_ref: str | None,
    fp_pos_markers: str | None,
    fp_neg_markers: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    bidcell_dir = output_dir / "bidcell"
    seg_dir = output_dir / "segmentation"
    bidcell_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # ── Generate config ───────────────────────────────────────────────────────
    print(f"\nGenerating BIDCell config...")
    config_path = write_config(
        output_dir=output_dir,
        xenium_dir=xenium_dir,
        bidcell_dir=bidcell_dir,
        cpus=cpus,
        total_steps=total_steps,
        total_epochs=total_epochs,
        fp_ref=fp_ref,
        fp_pos_markers=fp_pos_markers,
        fp_neg_markers=fp_neg_markers,
    )

    # ── Step 1: Run BIDCell (preprocess + train + predict) ────────────────────
    run_step(
        [py, HERE / "run_bidcell.py", "--config", config_path],
        logs_dir / "01_run_bidcell.log",
        "Step 1/3 — Run BIDCell (preprocess → train → predict)",
    )

    # ── Step 2: Convert mask to Xenium coordinates ────────────────────────────
    mask_path = find_latest_mask(bidcell_dir)
    print(f"\n  Found BIDCell mask: {mask_path}")
    run_step(
        [py, HERE / "to_xenium.py",
         "--mask",        mask_path,
         "--transcripts", xenium_dir / "transcripts.parquet",
         "--offsets",     offsets,
         "--output-dir",  seg_dir],
        logs_dir / "02_to_xenium.log",
        "Step 2/3 — Convert mask to Xenium coordinates",
    )

    # ── Step 3: Build GeoJSON for xeniumranger ────────────────────────────────
    geojson_path = output_dir / "cells.geojson"
    run_step(
        [py, REPO / "converters" / "make_geojson.py",
         "--boundaries", seg_dir / "cell_boundaries.parquet",
         "--output",     geojson_path],
        logs_dir / "03_make_geojson.log",
        "Step 3/3 — Build GeoJSON",
    )

    print(f"\n{'='*60}")
    print("  BIDCell pipeline complete!")
    print(f"{'='*60}")
    print(f"  BIDCell working dir:  {bidcell_dir}/")
    print(f"  Segmentation:         {seg_dir}/")
    print(f"  GeoJSON:              {geojson_path}")
    print(f"  Logs:                 {logs_dir}/")
    print()
    print("  Next step — import into Xenium Explorer:")
    print(f"    python xeniumranger/import_segmentation.py bidcell \\")
    print(f"        --id experiment_bidcell \\")
    print(f"        --xenium-bundle /path/to/outs_subset_bundle \\")
    print(f"        --cells {geojson_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--xenium-dir", required=True, type=Path,
                        help="Path to outs_subset/ directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write all BIDCell outputs")
    parser.add_argument("--offsets", type=Path, default=None,
                        help="Path to subset_offsets.json "
                             "(default: <xenium-dir>/subset_offsets.json)")
    parser.add_argument("--cpus", type=int, default=4,
                        help="CPUs for BIDCell multiprocessing (default: 4)")
    parser.add_argument("--total-steps", type=int, default=4000,
                        help="BIDCell training steps (default: 4000)")
    parser.add_argument("--total-epochs", type=int, default=1,
                        help="BIDCell training epochs (default: 1)")
    parser.add_argument("--fp-ref", default=None,
                        help="Path to reference cell-type CSV (optional)")
    parser.add_argument("--fp-pos-markers", default=None,
                        help="Path to positive markers CSV (optional)")
    parser.add_argument("--fp-neg-markers", default=None,
                        help="Path to negative markers CSV (optional)")
    args = parser.parse_args()

    offsets = args.offsets or (args.xenium_dir / "subset_offsets.json")
    if not offsets.exists():
        parser.error(f"subset_offsets.json not found at {offsets}. "
                     "Pass --offsets or ensure xenium-dir contains subset_offsets.json.")

    run_pipeline(
        xenium_dir=args.xenium_dir,
        output_dir=args.output_dir,
        offsets=offsets,
        cpus=args.cpus,
        total_steps=args.total_steps,
        total_epochs=args.total_epochs,
        fp_ref=args.fp_ref,
        fp_pos_markers=args.fp_pos_markers,
        fp_neg_markers=args.fp_neg_markers,
    )


if __name__ == "__main__":
    main()
