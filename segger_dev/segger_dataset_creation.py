from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Optional: provide transcript embeddings (rows: genes, cols: embedding dims)
# For example, cell-type abundance embeddings indexed by gene name
# weights = pd.DataFrame(..., index=gene_names)
weights = None  # set to a DataFrame if available

from segger.data.parquet.sample import STSampleParquet

base_dir = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/outs")
data_dir = Path("/data-hdd0/Ethan_Baird/Dec25_xenium/processed_tiles")

sample = STSampleParquet(
    base_dir=base_dir,
    sample_type="xenium",
    n_workers=4,            # controls parallel tiling across regions
    # weights=weights,        # optional transcript embeddings
    scale_factor=1.0,       # optional override (geometry scaling)
)

# Save tiles (choose either tile_size OR tile_width+tile_height)
sample.save(
    data_dir=data_dir,
    # Receptive fields (neighbors)
    k_bd=3,        # nearest boundaries per transcript
    dist_bd=15.0,  # max distance for tx->bd neighbors (µm-equivalent)
    k_tx=20,       # nearest transcripts per transcript
    dist_tx=5.0,   # max distance for tx->tx neighbors
    # Optional broader receptive fields for mutually exclusive genes (if used)
    # Tiling
    tile_size=50000,   # alternative: tile_width=..., tile_height=...
    # Sampling/splitting
    neg_sampling_ratio=5.0,
    frac=1.0,
    val_prob=0.1,
    test_prob=0.2,
)