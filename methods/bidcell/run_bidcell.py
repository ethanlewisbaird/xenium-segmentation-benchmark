"""
Run BIDCell segmentation.

Requires BIDCell to be installed: pip install bidcell
Requires a config YAML with paths to your data. See config_template.yaml.

Usage:
  python methods/bidcell/run_bidcell.py --config methods/bidcell/config_template.yaml
"""

import argparse


def run_bidcell(config_path: str) -> None:
    from bidcell import BIDCellModel

    # Preprocessing (nuclei segmentation, expression maps, patches, cell-gene
    # matrix) must be done before training. Set cc_weight/pos_weight/neg_weight
    # to 0 in config to disable reference-based losses if no reference data.
    model = BIDCellModel(config_path)
    model.train()
    model.predict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--config", required=True,
        help="Path to BIDCell config YAML (see config_template.yaml)",
    )
    args = parser.parse_args()
    run_bidcell(args.config)


if __name__ == "__main__":
    main()
