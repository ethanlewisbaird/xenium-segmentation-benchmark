"""
Create a single benchmark bundle containing multiple segmentation methods.

All large shared files (transcripts.zarr.zip, morphology, aux_outputs) are
stored once using hardlinks. Per-method files go in named subdirectories.
Each method gets its own experiment_<method>.xenium that references the
right subdirectory.

Structure:
  benchmark_bundle/
  ├── experiment_<method>.xenium   (one per method)
  ├── transcripts.zarr.zip         (shared, hardlinked, 3.3 GB)
  ├── transcripts.parquet          (shared, hardlinked, 2.0 GB)
  ├── morphology_focus/            (shared, hardlinked, 1.9 GB)
  ├── aux_outputs/                 (shared, hardlinked, 271 MB)
  ├── gene_panel.json              (shared, copied)
  └── <method>/                    (per-method unique files)
      ├── cells.zarr.zip
      ├── cell_feature_matrix.zarr.zip
      ├── analysis.zarr.zip
      ├── analysis_summary.html
      ├── cells.parquet / cells.csv.gz
      ├── cell_boundaries.parquet / .csv.gz
      ├── cell_feature_matrix.h5 / dir
      ├── analysis/ dir
      ├── metrics_summary.csv
      ├── nucleus_boundaries.*
      └── cell_id_map.csv.gz

Usage:
  python create_benchmark_bundle.py                     # build from scratch
  python create_benchmark_bundle.py --add <method> <outs_dir>  # add a new method

Xenium Explorer: open any experiment_<method>.xenium file in the bundle dir.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

BUNDLE = Path("benchmark_bundle")

# Per-method file names that belong in the method subdirectory
PER_METHOD_FILES = [
    "cells.zarr.zip",
    "cells.parquet",
    "cells.csv.gz",
    "cell_boundaries.parquet",
    "cell_boundaries.csv.gz",
    "cell_feature_matrix.zarr.zip",
    "cell_feature_matrix.h5",
    "analysis.zarr.zip",
    "analysis_summary.html",
    "metrics_summary.csv",
    "nucleus_boundaries.parquet",
    "nucleus_boundaries.csv.gz",
    "cell_id_map.csv.gz",
]
PER_METHOD_DIRS = ["cell_feature_matrix", "analysis"]

# Shared files / dirs (hardlinked from source bundle, stored once)
SHARED_FILES = ["gene_panel.json", "transcripts.parquet", "transcripts.zarr.zip"]
SHARED_DIRS  = ["morphology_focus", "aux_outputs"]


def hardlink_or_copy(src: Path, dst: Path):
    """Try hardlink first (zero extra disk); fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def hardlink_dir(src: Path, dst: Path):
    """Recursively hardlink (or copy) a directory tree."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            hardlink_or_copy(item, target)


def make_method_xenium(src_xenium: Path, method: str, bundle_dir: Path) -> dict:
    """
    Build experiment.xenium for a method inside the bundle.

    Rewrites cells_zarr, cell_features_zarr, analysis_zarr, analysis_summary
    to point into <method>/.  Transcripts and morphology stay at root.
    """
    data = json.loads(src_xenium.read_text())

    xef = data.get("xenium_explorer_files", {})
    xef["transcripts_zarr_filepath"]  = "transcripts.zarr.zip"
    xef["cells_zarr_filepath"]        = f"{method}/cells.zarr.zip"
    xef["cell_features_zarr_filepath"]= f"{method}/cell_feature_matrix.zarr.zip"
    xef["analysis_zarr_filepath"]     = f"{method}/analysis.zarr.zip"
    xef["analysis_summary_filepath"]  = f"{method}/analysis_summary.html"
    data["xenium_explorer_files"] = xef

    # Keep morphology_focus path unchanged (it's at root)
    return data


def add_method(method: str, src_dir: Path, bundle_dir: Path, is_first: bool = False):
    """
    Add one segmentation method's files into the bundle.

    If `is_first`, also copies/links the shared files.
    """
    src_dir = src_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    method_dir = bundle_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Adding method: {method} ===")
    print(f"  Source: {src_dir}")
    print(f"  Destination: {bundle_dir}")

    # ── Shared files (only on first method, already exist otherwise) ──────────
    if is_first:
        print("  Linking shared files...")
        for fname in SHARED_FILES:
            src = src_dir / fname
            dst = bundle_dir / fname
            if src.exists() and not dst.exists():
                print(f"    {fname}")
                hardlink_or_copy(src, dst)
            elif not src.exists():
                print(f"    WARNING: {fname} not found in source, skipping")

        for dname in SHARED_DIRS:
            src = src_dir / dname
            dst = bundle_dir / dname
            if src.exists() and not dst.exists():
                print(f"    {dname}/")
                hardlink_dir(src, dst)
            elif not src.exists():
                print(f"    WARNING: {dname}/ not found in source, skipping")

    # ── Per-method files ───────────────────────────────────────────────────────
    print(f"  Linking per-method files to {method}/...")
    for fname in PER_METHOD_FILES:
        src = src_dir / fname
        dst = method_dir / fname
        if src.exists():
            hardlink_or_copy(src, dst)
        # else silently skip (e.g. nucleus_boundaries can be empty)

    for dname in PER_METHOD_DIRS:
        src = src_dir / dname
        dst = method_dir / dname
        if src.exists() and not dst.exists():
            hardlink_dir(src, dst)

    # ── Write experiment_<method>.xenium ─────────────────────────────────────
    src_xenium = src_dir / "experiment.xenium"
    if not src_xenium.exists():
        print(f"  WARNING: experiment.xenium not found in {src_dir}")
        return

    xenium_data = make_method_xenium(src_xenium, method, bundle_dir)
    out_xenium = bundle_dir / f"experiment_{method}.xenium"
    out_xenium.write_text(json.dumps(xenium_data, indent=2))
    print(f"  Wrote {out_xenium.name}")

    print(f"  Done: {method}")


def build_from_scratch(methods: list[tuple[str, Path]], bundle_dir: Path):
    """Build the benchmark bundle from the given method outs directories."""
    for i, (method, src_dir) in enumerate(methods):
        if not src_dir.exists():
            print(f"WARNING: {src_dir} not found, skipping {method}")
            continue
        add_method(method, src_dir, bundle_dir, is_first=(i == 0))

    print(f"\n=== Bundle created at: {bundle_dir.resolve()} ===")
    print_summary(bundle_dir)


def print_summary(bundle_dir: Path):
    if not bundle_dir.exists():
        return
    total = 0
    for f in bundle_dir.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    print(f"Total (logical): {total/1e9:.1f} GB")
    result = os.popen(f"du -sh {bundle_dir} 2>/dev/null").read().split()[0]
    print(f"Disk usage (actual): {result}")
    print()
    print("Contents:")
    for p in sorted(bundle_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name}  ({p.stat().st_size/1e6:.0f} MB)")
        else:
            size = int(os.popen(f"du -sk {p} 2>/dev/null").read().split()[0]) * 1024
            print(f"  {p.name}/  ({size/1e6:.0f} MB)")
    print()
    print("To open in Xenium Explorer, open any of:")
    for xf in sorted(bundle_dir.glob("experiment_*.xenium")):
        print(f"  {xf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-dir", type=Path, default=BUNDLE,
                        help=f"Bundle output directory (default: {BUNDLE})")
    parser.add_argument("--add", nargs=2, metavar=("METHOD", "OUTS_DIR"),
                        help="Add a new method to an existing bundle")
    parser.add_argument("--methods", nargs="+", metavar="METHOD=OUTS_DIR",
                        help="Methods to bundle, e.g. bidcell=experiment_bidcell/outs "
                             "segger=experiment_segger/outs")
    args = parser.parse_args()

    bundle_dir = args.output_dir

    if args.add:
        method, outs_dir = args.add
        add_method(method, Path(outs_dir), bundle_dir, is_first=not bundle_dir.exists())
        print_summary(bundle_dir)
    elif args.methods:
        method_list = []
        for item in args.methods:
            name, path = item.split("=", 1)
            method_list.append((name, Path(path)))
        build_from_scratch(method_list, bundle_dir)
    else:
        # Default: look for experiment_bidcell/outs and experiment_segger/outs
        # relative to the current working directory
        cwd = Path.cwd()
        default_methods = [
            ("bidcell", cwd / "experiment_bidcell" / "outs"),
            ("segger",  cwd / "experiment_segger"  / "outs"),
        ]
        build_from_scratch(default_methods, bundle_dir)
