"""
Run the full BIDCell pipeline: preprocess → train → predict.

Requires BIDCell to be installed: pip install bidcell
Requires a config YAML with paths to your data. See config_template.yaml.

If fp_ref is not set in the config (no reference cell types available),
preannotation is skipped and all nuclei are assigned type 0 (Unknown).
The reference-based losses (cc_weight, pos_weight, neg_weight) should be
set to 0.0 in the config when running without reference data.

Usage:
  python methods/bidcell/run_bidcell.py --config methods/bidcell/config_template.yaml
"""

import argparse
import os
from pathlib import Path


def make_dummy_preannotation(model) -> None:
    """
    Create a dummy nuclei_cell_type.h5 with all nuclei assigned type 0 (Unknown).

    Called when no reference cell-type data is available. Reads the
    cell-gene matrix to get cell IDs, then writes the h5 file that
    the training step expects.
    """
    import glob
    import numpy as np
    import pandas as pd
    import h5py

    config = model.config
    data_dir = config.files.data_dir
    expr_dir = os.path.join(data_dir, config.files.dir_cgm, "nuclei")
    expr_file = os.path.join(expr_dir, config.files.fp_expr)

    if not os.path.exists(expr_file):
        raise FileNotFoundError(
            f"Cell-gene matrix not found at {expr_file}. "
            "Make sure segment_nuclei, generate_expression_maps, "
            "generate_patches, and make_cell_gene_mat have run first."
        )

    df_cells = pd.read_csv(expr_file, index_col=0)
    n_cells = df_cells.shape[0]
    print(f"  Creating dummy preannotation for {n_cells} nuclei (all type 0 = Unknown)")

    cell_id_col = df_cells.index.to_numpy()
    cell_type_col = np.zeros(n_cells, dtype=np.int32)

    h5_path = os.path.join(data_dir, config.files.fp_nuclei_anno)
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("data", data=cell_type_col)
        h5f.create_dataset("ids", data=cell_id_col)

    print(f"  Written: {h5_path}")


def run_bidcell(config_path: str) -> None:
    from bidcell import BIDCellModel

    model = BIDCellModel(config_path)
    config = model.config
    has_reference = bool(getattr(config.files, "fp_ref", None))

    if has_reference:
        print("Reference data found — running full pipeline with preannotation.")
        model.run_pipeline()
    else:
        print("No reference data — running pipeline with dummy preannotation.")
        print("\n### Preprocessing ###")
        model.segment_nuclei()
        model.generate_expression_maps()
        model.generate_patches()
        model.make_cell_gene_mat()

        print("\n### Preannotation (dummy — no reference) ###")
        make_dummy_preannotation(model)

        print("\n### Training ###")
        model.train()

        print("\n### Predict ###")
        model.predict()

        print("\n### Done ###")


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
