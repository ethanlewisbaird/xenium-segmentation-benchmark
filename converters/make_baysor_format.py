"""
Convert Segger transcript parquet → Baysor-format CSV for xeniumranger.

xeniumranger --transcript-assignment expects:
  transcript_id (uint64), cell (str like "seg-N" or "" unassigned), is_noise (bool)

Usage:
  python make_baysor_format.py \
      --segger-parquet segger_output/.../segger_transcripts.parquet \
      --output-csv     segger_transcript_assignment.csv \
      [--min-tx 5]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_baysor_csv(parquet_path: Path, output_csv: Path, min_tx: int = 5) -> None:
    print("Loading transcripts...")
    df = pd.read_parquet(parquet_path, columns=["transcript_id", "segger_cell_id"])
    print(f"  {len(df):,} rows")

    counts = df.groupby("segger_cell_id").size()
    assigned_mask = df["segger_cell_id"] != "UNASSIGNED"
    keep_cells = counts[(counts >= min_tx) & (counts.index != "UNASSIGNED")].index
    print(f"  Cells with >= {min_tx} transcripts: {len(keep_cells):,}")

    # Sequential integer IDs 1..N
    cell_map = {cid: f"seg-{i + 1}" for i, cid in enumerate(keep_cells)}

    print("Building transcript assignment CSV...")
    df["cell"] = df["segger_cell_id"].map(cell_map).fillna("")
    df["is_noise"] = (~assigned_mask | ~df["segger_cell_id"].isin(keep_cells)).astype(int)
    df["transcript_id"] = df["transcript_id"].astype(np.int64)

    out = df[["transcript_id", "cell", "is_noise"]]
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}  ({len(out):,} rows, {output_csv.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--segger-parquet", required=True,
        help="Path to segger_transcripts.parquet",
    )
    parser.add_argument(
        "--output-csv", required=True,
        help="Output CSV path for xeniumranger --transcript-assignment",
    )
    parser.add_argument(
        "--min-tx", type=int, default=5,
        help="Minimum transcripts per cell to include (default: 5)",
    )
    args = parser.parse_args()
    make_baysor_csv(Path(args.segger_parquet), Path(args.output_csv), args.min_tx)


if __name__ == "__main__":
    main()
