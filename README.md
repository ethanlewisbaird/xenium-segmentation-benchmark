# xenium-segmentation-benchmark

A benchmarking framework for cell segmentation methods on 10x Genomics Xenium spatial transcriptomics data.

Wraps multiple segmentation methods (currently Segger and BIDCell), converts their outputs to xeniumranger-compatible formats, and creates a unified Xenium Explorer bundle so all methods can be compared side-by-side.

## Architecture

```
outs_subset/                    ← subset directory (from xenium-subsetter)
outs_subset_bundle/             ← xeniumranger-compatible bundle (from xenium-subsetter)
      │
      ├── methods/segger/run_pipeline.py   (or individual: create_dataset → train → run_predict → to_xenium)
      ├── methods/bidcell/run_pipeline.py  (or individual: run_bidcell → to_xenium)
      │
      ▼
segmentation outputs (per-method cell boundaries + transcript assignments)
      │
      ├── converters/
      │     make_geojson.py        cell_boundaries.parquet → GeoJSON polygons
      │     make_baysor_format.py  Segger parquet → Baysor-format CSV for xeniumranger
      │     make_viz_polygons.py   Segger boundaries → GeoJSON viz polygons
      │
      ▼
xeniumranger/import_segmentation.py  (one run per method)
      │
      ▼
experiment_<method>/outs/       xeniumranger output per method
      │
bundle/create_benchmark_bundle.py
      │
      ▼
benchmark_bundle/               single download, all methods
      ├── experiment_<method>.xenium   (one per method)
      ├── transcripts.zarr.zip         (shared, hardlinked)
      ├── morphology_focus/            (shared, hardlinked)
      ├── aux_outputs/                 (shared, hardlinked)
      └── <method>/                    (per-method files)
```

## Supported Methods

| Method | Type | Environment | Output format |
|--------|------|-------------|---------------|
| [Segger](https://github.com/EliHei2/segger_dev) | GNN link prediction | `segger311` conda | Parquet transcript assignments |
| [BIDCell](https://github.com/SydneyBioX/BIDCell) | Self-supervised DL | `bidcell` conda | Segmentation mask (TIF) |

## Prerequisites

- [xeniumranger](https://www.10xgenomics.com/support/software/xenium-ranger) (≥ 4.0) installed and on `$PATH` (or pass `--xeniumranger /path/to/xeniumranger`)
- A Xenium subset created by [xenium-subsetter](https://github.com/ethanlewisbaird/xenium-subsetter)

## Installation

```bash
git clone https://github.com/ethanlewisbaird/xenium-segmentation-benchmark
cd xenium-segmentation-benchmark
pip install -r requirements.txt
```

### Segger environment

Segger source is vendored in `segger_dev/`. Install it with its dependencies:

```bash
conda create -n segger311 python=3.11
conda activate segger311
pip install torch torchvision torch_geometric
pip install -e segger_dev/
```

### BIDCell environment

```bash
conda create -n bidcell python=3.10
conda activate bidcell
pip install bidcell
```

## Usage

### 0. Prepare input data

Use [xenium-subsetter](https://github.com/ethanlewisbaird/xenium-subsetter) to create a subset directory and bundle from your Xenium dataset:

```bash
python -m xenium_subsetter.subset \
    --xenium-dir  /path/to/outs \
    --coords-dir  /path/to/selection_coordinates \
    --output-dir  /path/to/outs_subset

python -m xenium_subsetter.build_bundle \
    --subset-dir  /path/to/outs_subset \
    --xenium-dir  /path/to/outs \
    --output-dir  /path/to/outs_subset_bundle
```

### 1. Run Segger

The pipeline runner handles all steps (create_dataset → train → predict → to_xenium → converters) and writes outputs to a structured directory with per-step log files:

```bash
conda run -n segger311 python methods/segger/run_pipeline.py \
    --xenium-dir /path/to/outs_subset \
    --output-dir /path/to/segger_output \
    [--epochs 100] [--batch-size 1] [--devices 1]
```

Output structure:
```
segger_output/
├── tiles/                     graph tiles
├── model/                     Lightning checkpoints
├── predict/                   Segger output parquet
├── segmentation/              cell_boundaries.parquet, cells.parquet
├── transcript_assignment.csv
├── viz_polygons.geojson
└── logs/
    ├── 01_create_dataset.log
    ├── 02_train.log
    ├── 03_run_predict.log
    ├── 04_to_xenium.log
    ├── 05a_make_baysor_format.log
    └── 05b_make_viz_polygons.log
```

<details>
<summary>Run individual steps manually</summary>

```bash
# Step 1 — Build graph tiles
conda run -n segger311 python methods/segger/create_dataset.py \
    --xenium-dir /path/to/outs_subset \
    --output-dir /path/to/segger_output/tiles

# Step 2 — Train the GNN
conda run -n segger311 python methods/segger/train.py \
    --tiles-dir  /path/to/segger_output/tiles \
    --output-dir /path/to/segger_output/model \
    --epochs 100 --devices 1

# Step 3 — Run inference
conda run -n segger311 python methods/segger/run_predict.py \
    --dataset-dir /path/to/segger_output/tiles \
    --output-dir  /path/to/segger_output/predict \
    --checkpoint  /path/to/segger_output/model/lightning_logs/version_0 \
    --transcripts /path/to/outs_subset/transcripts.parquet

# Step 4 — Convert to original Xenium coordinates
conda run -n segger311 python methods/segger/to_xenium.py \
    --segger-parquet /path/to/segger_output/predict/.../transcripts_df.parquet \
    --transcripts    /path/to/outs_subset/transcripts.parquet \
    --offsets        /path/to/outs_subset/subset_offsets.json \
    --output-dir     /path/to/segger_output/segmentation

# Step 5 — Convert to xeniumranger formats
conda run -n segger311 python converters/make_baysor_format.py \
    --segger-parquet /path/to/segger_output/predict/.../transcripts_df.parquet \
    --output-csv     /path/to/segger_output/transcript_assignment.csv

conda run -n segger311 python converters/make_viz_polygons.py \
    --boundaries     /path/to/segger_output/segmentation/cell_boundaries.parquet \
    --segger-parquet /path/to/segger_output/predict/.../transcripts_df.parquet \
    --output         /path/to/segger_output/viz_polygons.geojson
```
</details>

### 2. Run BIDCell

The pipeline runner auto-generates a config, runs BIDCell (preprocess → train → predict), converts the mask, and builds the GeoJSON:

```bash
conda run -n bidcell python methods/bidcell/run_pipeline.py \
    --xenium-dir /path/to/outs_subset \
    --output-dir /path/to/bidcell_output \
    [--cpus 4] [--total-steps 4000] [--total-epochs 1]

# With reference cell types (optional):
#   --fp-ref /path/to/reference.csv
#   --fp-pos-markers /path/to/pos_markers.csv
#   --fp-neg-markers /path/to/neg_markers.csv
```

Output structure:
```
bidcell_output/
├── config.yaml               auto-generated BIDCell config
├── bidcell/                  BIDCell working dir (model_outputs/ inside)
├── segmentation/             cell_boundaries.parquet, cells.parquet
├── cells.geojson
└── logs/
    ├── 01_run_bidcell.log
    ├── 02_to_xenium.log
    └── 03_make_geojson.log
```

<details>
<summary>Run individual steps manually</summary>

```bash
# Copy and edit the config template
cp methods/bidcell/config_template.yaml my_bidcell_config.yaml
# Edit: files.data_dir, files.fp_dapi, files.fp_transcripts

# Run BIDCell (preprocess → train → predict)
conda run -n bidcell python methods/bidcell/run_bidcell.py \
    --config my_bidcell_config.yaml

# Convert mask to original Xenium coordinates
conda run -n bidcell python methods/bidcell/to_xenium.py \
    --mask        /path/to/model_outputs/.../epoch_N_step_M.tif \
    --transcripts /path/to/outs_subset/transcripts.parquet \
    --offsets     /path/to/outs_subset/subset_offsets.json \
    --output-dir  /path/to/bidcell_output/segmentation

# Convert to GeoJSON for xeniumranger
conda run -n bidcell python converters/make_geojson.py \
    --boundaries /path/to/bidcell_output/segmentation/cell_boundaries.parquet \
    --output     /path/to/bidcell_output/cells.geojson
```
</details>

### 3. Import into Xenium Explorer via xeniumranger

Use the provided wrapper script (handles method-specific argument differences):

**BIDCell** (polygon-based):
```bash
python xeniumranger/import_segmentation.py bidcell \
    --id            experiment_bidcell \
    --xenium-bundle /path/to/outs_subset_bundle \
    --cells         /path/to/bidcell_cells.geojson \
    --localcores    8 --localmem 48
```

**Segger** (transcript-assignment-based):
```bash
python xeniumranger/import_segmentation.py segger \
    --id                    experiment_segger \
    --xenium-bundle         /path/to/outs_subset_bundle \
    --transcript-assignment /path/to/segger_transcript_assignment.csv \
    --viz-polygons          /path/to/segger_viz_polygons.geojson \
    --localcores            8 --localmem 48
```

Pass `--xeniumranger /path/to/xeniumranger` if it is not on `$PATH`.  
Pass `--nopreflight` to skip xeniumranger preflight checks (required when using a subset bundle whose morphology TIFF predates the UUID requirement in xeniumranger 4.x).

### 4. Build the benchmark bundle

```bash
python bundle/create_benchmark_bundle.py \
    --output-dir /path/to/benchmark_bundle \
    --methods \
        bidcell=/path/to/experiment_bidcell/outs \
        segger=/path/to/experiment_segger/outs
```

Add a new method to an existing bundle:
```bash
python bundle/create_benchmark_bundle.py \
    --output-dir /path/to/benchmark_bundle \
    --add <method> /path/to/experiment_<method>/outs
```

Open any `experiment_*.xenium` file in Xenium Explorer.

**Download the bundle to another machine:**
```bash
rsync -avz server:/path/to/benchmark_bundle/ ./benchmark_bundle/
```

## Key design decisions

### BIDCell uses `--cells` (polygon import)
BIDCell outputs a segmentation mask. We convert it to per-cell convex hull polygons in original Xenium coordinates and pass them to xeniumranger's `--cells` option. xeniumranger performs spatial transcript assignment.

### Segger uses `--transcript-assignment` (Baysor CSV import)
Segger directly assigns transcripts to cells. For large datasets (100k+ cells, 50M+ transcripts), the `--cells` approach causes xeniumranger's ambiguity-resolution stage to OOM. The `--transcript-assignment` approach bypasses spatial assignment entirely, using Segger's pre-computed assignments.

### No-reference BIDCell
When no reference cell-type CSV is available, `run_bidcell.py` automatically creates dummy placeholder files for BIDCell's Pydantic config (which requires these fields), sets all cells to type "Unknown", and zeroes the reference-based loss weights. No manual config workarounds needed.

### Benchmark bundle uses hardlinks
Large shared files (`transcripts.zarr.zip`, `morphology_focus/`, `aux_outputs/`) are stored once via OS hardlinks. Each new method adds only ~100–300 MB.

### Coordinate system
All segmentation tool outputs are in **subset-relative coordinates** (origin = ROI top-left). The `to_xenium.py` scripts in each method directory add `x_offset_um` / `y_offset_um` from `subset_offsets.json` to convert back to original Xenium space before xeniumranger import.

## Adding a New Segmentation Method

1. Create `methods/<method>/` with a script that outputs cell boundaries (parquet or mask) in subset-relative coordinates, plus a `to_xenium.py` that converts to original coordinates.
2. Add a converter in `converters/` if a new output format is needed.
3. Import via xeniumranger:
   ```bash
   python xeniumranger/import_segmentation.py <method> \
       --id experiment_<method> --xenium-bundle /path/to/outs_subset_bundle \
       [method-specific args]
   ```
4. Add to bundle:
   ```bash
   python bundle/create_benchmark_bundle.py \
       --output-dir /path/to/benchmark_bundle \
       --add <method> experiment_<method>/outs
   ```
