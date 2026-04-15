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
import tempfile
from pathlib import Path

import yaml


def _patch_config_for_no_reference(config_path: str) -> tuple[str, bool]:
    """
    Load the YAML config and check whether fp_ref is set.

    If fp_ref is absent or null, create three empty placeholder CSV files in
    a temporary directory and inject their paths into a copy of the config.
    Returns (path_to_use, has_reference).  When has_reference is False the
    returned path points to a NamedTemporaryFile that the caller owns.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    files = cfg.get("files", {})
    fp_ref = files.get("fp_ref") or None
    has_reference = bool(fp_ref)

    if has_reference:
        return config_path, True

    # BIDCell's Pydantic model requires fp_ref / fp_pos_markers / fp_neg_markers
    # to be non-null strings that point to existing files — even when the
    # reference-based losses are set to 0 and preannotate() is never called.
    # Create empty placeholder files so Pydantic validation passes.
    placeholder_dir = Path(tempfile.mkdtemp(prefix="bidcell_placeholders_"))
    for key in ("fp_ref", "fp_pos_markers", "fp_neg_markers"):
        p = placeholder_dir / f"{key}_placeholder.csv"
        p.touch()
        files[key] = str(p)

    cfg["files"] = files

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="bidcell_cfg_"
    )
    yaml.dump(cfg, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name, False


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
    config_to_use, has_reference = _patch_config_for_no_reference(config_path)

    try:
        from bidcell import BIDCellModel

        model = BIDCellModel(config_to_use)

        if has_reference:
            print("Reference data found — running full pipeline with preannotation.")
            model.run_pipeline()
        else:
            print("No reference data — running pipeline with dummy preannotation.")
            print("\n### Preprocessing ###")
            model.segment_nuclei()
            model.generate_expression_maps()
            model.generate_patches()
            model.make_cell_gene_mat(is_cell=False)

            print("\n### Preannotation (dummy — no reference) ###")
            make_dummy_preannotation(model)

            print("\n### Training ###")
            model.train()

            print("\n### Predict ###")
            model.predict()

            print("\n### Done ###")
    finally:
        if config_to_use != config_path:
            os.unlink(config_to_use)


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
