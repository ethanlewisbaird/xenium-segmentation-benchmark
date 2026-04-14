"""
Create Segger graph tiles from Xenium data.

Reads transcripts.parquet and cell_boundaries.parquet from the Xenium
(or subset) directory, builds a heterogeneous graph dataset, and saves
train/val/test tiles for Segger training and inference.

Usage:
  python methods/segger/create_dataset.py \
      --xenium-dir  outs_subset/ \
      --output-dir  segger_tiles/ \
      [--k-bd 3] [--dist-bd 15.0] \
      [--k-tx 20] [--dist-tx 5.0] \
      [--tile-size 50000] \
      [--val-prob 0.1] [--test-prob 0.2] \
      [--workers 4]
"""

import argparse
import sys
from pathlib import Path


def create_dataset(
    xenium_dir: Path,
    output_dir: Path,
    k_bd: int = 3,
    dist_bd: float = 15.0,
    k_tx: int = 20,
    dist_tx: float = 5.0,
    tile_size: int = 50000,
    val_prob: float = 0.1,
    test_prob: float = 0.2,
    workers: int = 4,
    sample_type: str = "xenium",
) -> None:
    # Import here so the module is importable without segger installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segger_dev" / "src"))
    from segger.data.parquet.sample import STSampleParquet

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building graph tiles from: {xenium_dir}")
    print(f"Output directory: {output_dir}")

    sample = STSampleParquet(
        base_dir=xenium_dir,
        sample_type=sample_type,
        n_workers=workers,
    )

    sample.save(
        data_dir=output_dir,
        k_bd=k_bd,
        dist_bd=dist_bd,
        k_tx=k_tx,
        dist_tx=dist_tx,
        tile_size=tile_size,
        neg_sampling_ratio=5.0,
        frac=1.0,
        val_prob=val_prob,
        test_prob=test_prob,
    )

    print(f"\nDone. Tiles saved to: {output_dir}")
    for split in ["train_tiles", "val_tiles", "test_tiles"]:
        d = output_dir / split / "processed"
        if d.exists():
            n = len(list(d.glob("*.pt")))
            print(f"  {split}: {n} tiles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--xenium-dir", required=True,
        help="Path to Xenium (or outs_subset) directory with transcripts.parquet "
             "and cell_boundaries.parquet",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save train/val/test graph tiles",
    )
    parser.add_argument("--k-bd", type=int, default=3,
                        help="Nearest boundaries per transcript (default: 3)")
    parser.add_argument("--dist-bd", type=float, default=15.0,
                        help="Max tx→boundary distance µm (default: 15.0)")
    parser.add_argument("--k-tx", type=int, default=20,
                        help="Nearest transcripts per transcript (default: 20)")
    parser.add_argument("--dist-tx", type=float, default=5.0,
                        help="Max tx→tx distance µm (default: 5.0)")
    parser.add_argument("--tile-size", type=int, default=50000,
                        help="Tile size in number of transcripts (default: 50000)")
    parser.add_argument("--val-prob", type=float, default=0.1,
                        help="Fraction of tiles for validation (default: 0.1)")
    parser.add_argument("--test-prob", type=float, default=0.2,
                        help="Fraction of tiles for test (default: 0.2)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for tiling (default: 4)")
    parser.add_argument("--sample-type", default="xenium",
                        help="Platform type: xenium, merscope, cosmx (default: xenium)")
    args = parser.parse_args()

    create_dataset(
        xenium_dir=Path(args.xenium_dir),
        output_dir=Path(args.output_dir),
        k_bd=args.k_bd,
        dist_bd=args.dist_bd,
        k_tx=args.k_tx,
        dist_tx=args.dist_tx,
        tile_size=args.tile_size,
        val_prob=args.val_prob,
        test_prob=args.test_prob,
        workers=args.workers,
        sample_type=args.sample_type,
    )


if __name__ == "__main__":
    main()
