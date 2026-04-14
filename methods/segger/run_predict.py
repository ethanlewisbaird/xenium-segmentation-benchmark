"""
Run Segger cell segmentation inference on pre-built graph tiles.

Usage:
  python methods/segger/run_predict.py \
      --dataset-dir /path/to/segger_tiles \
      --output-dir  /path/to/segger_output \
      --checkpoint  /path/to/segger_model/lightning_logs/version_N \
      --transcripts /path/to/outs_subset/transcripts.parquet \
      [--seg-tag segger_run]
"""

import argparse
import glob
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def run_predict(
    dataset_dir: Path,
    output_dir: Path,
    model_version_dir: Path,
    transcripts: Path,
    seg_tag: str = "segger",
) -> None:
    import torch
    from torch_geometric.nn import to_hetero

    from segger.models.segger_model import Segger
    from segger.prediction.predict_parquet import segment
    from segger.training.segger_data_module import SeggerDataModule
    from segger.training.train import LitSegger

    dm = SeggerDataModule(data_dir=str(dataset_dir), batch_size=1, num_workers=2)
    dm.setup()
    logger.info(f"Data loaded: {len(dm.train)} train tiles")

    ckpt_dir = model_version_dir / "checkpoints"
    ckpts = sorted(
        glob.glob(str(ckpt_dir / "*.ckpt")),
        key=lambda c: tuple(int(x) for x in re.findall(r"\d+", c)),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    ckpt_path = ckpts[-1]
    logger.info(f"Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    num_tx_tokens = state["model.tx_embedding.tx.weight"].shape[0]
    init_emb = state["model.tx_embedding.tx.weight"].shape[1]
    first_att_key = [k for k in state if "conv_first" in k and k.endswith(".att")][0]
    _, heads, hid_ch = state[first_att_key].shape
    last_att_key = [k for k in state if "conv_last" in k and k.endswith(".att")][0]
    _, _, out_ch = state[last_att_key].shape
    num_mid = len(
        set(
            k.split(".conv_mid_layers.")[1].split(".")[0]
            for k in state
            if ".conv_mid_layers." in k
        )
    )

    base_model = Segger(
        num_tx_tokens=num_tx_tokens,
        init_emb=init_emb,
        hidden_channels=hid_ch,
        out_channels=out_ch,
        heads=heads,
        num_mid_layers=num_mid,
    )
    metadata = dm.train[0].metadata()
    hetero_model = to_hetero(base_model, metadata=metadata, aggr="sum")
    lit = LitSegger(model=hetero_model, learning_rate=1e-3)
    lit.load_state_dict(state)
    lit.eval()
    logger.info("Model reconstructed OK")

    segment(
        lit,
        dm,
        save_dir=str(output_dir),
        seg_tag=seg_tag,
        transcript_file=str(transcripts),
        receptive_field={"k_bd": 4, "dist_bd": 12.0, "k_tx": 5, "dist_tx": 5.0},
        use_cc=False,
        knn_method="kd_tree",
        verbose=True,
    )
    logger.info("DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to pre-built segger graph tiles directory",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write Segger output parquet",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to Segger model version directory (containing checkpoints/)",
    )
    parser.add_argument(
        "--transcripts", required=True,
        help="Path to outs_subset/transcripts.parquet",
    )
    parser.add_argument(
        "--seg-tag", default="segger",
        help="Tag prefix for output directory name (default: segger)",
    )
    args = parser.parse_args()
    run_predict(
        Path(args.dataset_dir),
        Path(args.output_dir),
        Path(args.checkpoint),
        Path(args.transcripts),
        args.seg_tag,
    )


if __name__ == "__main__":
    main()
