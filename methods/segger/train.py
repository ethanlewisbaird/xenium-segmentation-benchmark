"""
Train the Segger GNN model on pre-built graph tiles.

Expects tiles produced by create_dataset.py (train_tiles/, val_tiles/ subdirs).
Saves Lightning checkpoints to output-dir/lightning_logs/version_N/checkpoints/.

Usage:
  python methods/segger/train.py \
      --tiles-dir   segger_tiles/ \
      --output-dir  segger_model/ \
      [--epochs 100] [--batch-size 4] \
      [--init-emb 8] [--hidden-channels 64] [--out-channels 16] [--heads 4] \
      [--lr 1e-3] [--devices 1] [--precision 16-mixed]
"""

import argparse
import sys
from pathlib import Path


def train(
    tiles_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 4,
    init_emb: int = 8,
    hidden_channels: int = 64,
    out_channels: int = 16,
    heads: int = 4,
    lr: float = 1e-3,
    devices: int = 1,
    accelerator: str = "cuda",
    precision: str = "16-mixed",
) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segger_dev" / "src"))

    import lightning as L
    from torch_geometric.nn import to_hetero
    from lightning.pytorch.plugins.environments import LightningEnvironment

    from segger.models.segger_model import Segger
    from segger.training.segger_data_module import SeggerDataModule
    from segger.training.train import LitSegger

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tiles from: {tiles_dir}")
    dm = SeggerDataModule(data_dir=tiles_dir, batch_size=batch_size, num_workers=2)
    dm.setup()
    print(f"  Train tiles: {len(dm.train)}")
    print(f"  Val tiles:   {len(dm.val)}")

    # Build model using metadata from first tile
    metadata = dm.train[0].metadata()
    base_model = Segger(
        num_tx_tokens=500,
        init_emb=init_emb,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
    )
    hetero_model = to_hetero(base_model, metadata=metadata, aggr="sum")
    lit = LitSegger(model=hetero_model, learning_rate=lr)

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=epochs,
        default_root_dir=str(output_dir),
        plugins=[LightningEnvironment()],
    )

    print(f"\nTraining for {epochs} epochs...")
    trainer.fit(lit, dm.train_dataloader(), dm.val_dataloader())
    print(f"\nDone. Checkpoints saved to: {output_dir}/lightning_logs/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--tiles-dir", required=True,
        help="Directory with train_tiles/ and val_tiles/ (from create_dataset.py)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save Lightning checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--init-emb", type=int, default=8,
                        help="Initial embedding size (default: 8)")
    parser.add_argument("--hidden-channels", type=int, default=64,
                        help="Hidden channels (default: 64)")
    parser.add_argument("--out-channels", type=int, default=16,
                        help="Output channels (default: 16)")
    parser.add_argument("--heads", type=int, default=4,
                        help="Attention heads (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--accelerator", default="cuda",
                        help="Accelerator: cuda or cpu (default: cuda)")
    parser.add_argument("--precision", default="16-mixed",
                        help="Precision: 16-mixed, 32 (default: 16-mixed)")
    args = parser.parse_args()

    train(
        tiles_dir=Path(args.tiles_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        init_emb=args.init_emb,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        heads=args.heads,
        lr=args.lr,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
