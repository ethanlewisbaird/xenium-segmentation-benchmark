"""
Run xeniumranger import-segmentation for a segmentation method.

Wraps xeniumranger's import-segmentation command with method-specific
argument handling.  Supports two import modes:

  bidcell (polygon-based):
    xeniumranger import-segmentation
        --id <id> --xenium-bundle <bundle> --cells <geojson> --units microns

  segger (transcript-assignment-based):
    xeniumranger import-segmentation
        --id <id> --xenium-bundle <bundle>
        --transcript-assignment <csv> --viz-polygons <geojson> --units microns

Usage:
  python xeniumranger/import_segmentation.py \\
      bidcell \\
      --id            experiment_bidcell \\
      --xenium-bundle outs_subset_bundle \\
      --cells         bidcell_cells.geojson \\
      [--localcores 8] [--localmem 48] [--xeniumranger /path/to/xeniumranger]

  python xeniumranger/import_segmentation.py \\
      segger \\
      --id                    experiment_segger \\
      --xenium-bundle         outs_subset_bundle \\
      --transcript-assignment segger_transcript_assignment.csv \\
      --viz-polygons          segger_viz_polygons.geojson \\
      [--localcores 8] [--localmem 48] [--xeniumranger /path/to/xeniumranger]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_xeniumranger(override: str | None) -> str:
    if override:
        return override
    found = shutil.which("xeniumranger")
    if found:
        return found
    raise FileNotFoundError(
        "xeniumranger not found on PATH. Install it from https://www.10xgenomics.com/support/software/xenium-ranger "
        "or pass --xeniumranger /path/to/xeniumranger."
    )


def run_import(args) -> None:
    xr = find_xeniumranger(args.xeniumranger)

    cmd = [
        xr, "import-segmentation",
        "--id",            args.id,
        "--xenium-bundle", str(args.xenium_bundle),
        "--units",         "microns",
        "--localcores",    str(args.localcores),
        "--localmem",      str(args.localmem),
    ]

    if args.method == "bidcell":
        cmd += ["--cells", str(args.cells)]
    elif args.method == "segger":
        cmd += [
            "--transcript-assignment", str(args.transcript_assignment),
            "--viz-polygons",          str(args.viz_polygons),
        ]

    if args.nopreflight:
        cmd += ["--nopreflight", "1"]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("method", choices=["bidcell", "segger"],
                        help="Segmentation method")
    parser.add_argument("--id", required=True,
                        help="Output experiment ID (xeniumranger --id)")
    parser.add_argument("--xenium-bundle", required=True, type=Path,
                        help="Path to xeniumranger input bundle")
    parser.add_argument("--localcores", type=int, default=8,
                        help="Number of CPU cores (default: 8)")
    parser.add_argument("--localmem", type=int, default=48,
                        help="Memory in GB (default: 48)")
    parser.add_argument("--xeniumranger", default=None,
                        help="Path to xeniumranger binary (default: from PATH)")
    parser.add_argument("--nopreflight", action="store_true",
                        help="Skip xeniumranger preflight checks (e.g. OME-TIFF UUID validation)")

    # BIDCell-specific
    parser.add_argument("--cells", type=Path, default=None,
                        help="[bidcell] GeoJSON cell polygons")

    # Segger-specific
    parser.add_argument("--transcript-assignment", type=Path, default=None,
                        help="[segger] Baysor-format transcript assignment CSV")
    parser.add_argument("--viz-polygons", type=Path, default=None,
                        help="[segger] GeoJSON viz polygons")

    args = parser.parse_args()

    if args.method == "bidcell" and not args.cells:
        parser.error("--cells is required for bidcell")
    if args.method == "segger" and not args.transcript_assignment:
        parser.error("--transcript-assignment is required for segger")
    if args.method == "segger" and not args.viz_polygons:
        parser.error("--viz-polygons is required for segger")

    run_import(args)


if __name__ == "__main__":
    main()
