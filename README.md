# xenium-benchmark

A benchmarking framework for cell segmentation methods on 10x Genomics Xenium spatial transcriptomics data.

Wraps multiple segmentation methods (currently Segger and BIDCell), converts their outputs to xeniumranger-compatible formats, and creates a unified Xenium Explorer bundle so all methods can be compared in a single download.

## Architecture

```
outs_subset/                    ← subset bundle (from xenium-subsetter)
      │
      ├── methods/segger/        run_predict.py
      ├── methods/bidcell/       to_xenium.py
      │
      ▼
segmentation outputs (per-method cell boundaries + transcript assignments)
      │
      ├── converters/
      │     make_geojson.py      cell_boundaries.parquet → GeoJSON polygons
      │     make_baysor_format.py → Baysor-format CSV for xeniumranger
      │     make_viz_polygons.py  → fast GeoJSON viz polygons
      │
      ▼
xeniumranger import-segmentation (one run per method)
      │
      ▼
experiment_<method>/outs/       xeniumranger output per method
      │
bundle/create_benchmark_bundle.py
      │
      ▼
benchmark_bundle/               single download, all methods
      ├── experiment_bidcell.xenium
      ├── experiment_segger.xenium
      ├── transcripts.zarr.zip  (shared, 3.5 GB)
      ├── morphology_focus/     (shared, 1.9 GB)
      ├── bidcell/              method-specific files (~60 MB)
      └── segger/               method-specific files (~216 MB)
```

## Supported Methods

| Method | Type | Environment | Output format |
|--------|------|-------------|---------------|
| [Segger](https://github.com/EliHei2/segger_dev) | GNN link prediction | `segger311` conda | Parquet transcript assignments |
| [BIDCell](https://github.com/SydneyBioX/BIDCell) | Self-supervised DL | `bidcell` conda | Segmentation mask (TIF) |

## Installation

### Prerequisites

- [xeniumranger](https://www.10xgenomics.com/support/software/xenium-ranger) (≥ 4.0) installed and on `$PATH`
- Conda environments for each method (see their respective repos)

### Method-specific setup

**Segger:**
```bash
conda create -n segger311 python=3.11
pip install torch torchvision torch_geometric
cd methods/segger && pip install -e path/to/segger_dev
```

**BIDCell:**
```bash
conda create -n bidcell python=3.10
pip install bidcell
```

## Usage

### 1. Prepare the subset

Use [xenium-subsetter](../xenium-subsetter) to create `outs_subset/`.

### 2. Run Segger

```bash
# Train (or use pre-trained tiles)
conda run -n segger311 python methods/segger/run_predict.py \
    --dataset-dir /path/to/processed_tiles \
    --output-dir  /path/to/segger_output \
    --checkpoint  /path/to/segger.ckpt

# Convert output to xeniumranger format
conda run -n segger311 python converters/make_baysor_format.py \
    --segger-parquet segger_output/.../segger_transcripts.parquet \
    --output-csv     segger_transcript_assignment.csv

conda run -n segger311 python converters/make_viz_polygons.py \
    --boundaries outs_subset/segger_segmentation/cell_boundaries.parquet \
    --output     segger_viz_polygons.geojson
```

### 3. Run BIDCell

```bash
conda run -n bidcell bidcell \
    --config Dec25_config.yaml

conda run -n bidcell python methods/bidcell/to_xenium.py
conda run -n bidcell python converters/make_geojson.py \
    --boundaries outs_subset/bidcell_segmentation/cell_boundaries.parquet \
    --output     bidcell_cells.geojson
```

### 4. Import into Xenium Explorer via xeniumranger

**BIDCell** (polygon-based):
```bash
xeniumranger import-segmentation \
    --id            experiment_bidcell \
    --xenium-bundle outs_subset_bundle \
    --cells         bidcell_cells.geojson \
    --units         microns \
    --localcores    8 --localmem 48
```

**Segger** (transcript-assignment-based):
```bash
xeniumranger import-segmentation \
    --id                    experiment_segger \
    --xenium-bundle         outs_subset_bundle \
    --transcript-assignment segger_transcript_assignment.csv \
    --viz-polygons          segger_viz_polygons.geojson \
    --units                 microns \
    --localcores            8 --localmem 48
```

### 5. Build the benchmark bundle

```bash
python bundle/create_benchmark_bundle.py
# or add a new method later:
python bundle/create_benchmark_bundle.py --add <method> experiment_<method>/outs/
```

**Download the bundle:**
```bash
rsync -avz server:/path/to/benchmark_bundle/ ./benchmark_bundle/
```

**Add a new method after the initial download:**
```bash
rsync -avz server:/path/to/benchmark_bundle/<method>/ ./benchmark_bundle/<method>/
rsync -avz server:/path/to/benchmark_bundle/experiment_<method>.xenium ./benchmark_bundle/
```

Open any `experiment_*.xenium` file in Xenium Explorer.

## Key design decisions

### BIDCell uses `--cells` (polygon import)
BIDCell outputs a segmentation mask. We convert it to per-cell convex hull polygons in original Xenium coordinates and pass them as GeoJSON to xeniumranger's `--cells` option. xeniumranger then performs spatial transcript assignment.

### Segger uses `--transcript-assignment` (Baysor CSV import)
Segger directly assigns transcripts to cells. With 148k+ cells and 77M+ transcripts, the `--cells` approach causes xeniumranger's ambiguity-resolution stage to OOM. The `--transcript-assignment` approach bypasses spatial assignment entirely, using Segger's pre-computed assignments.

### Benchmark bundle uses hardlinks
The `benchmark_bundle/` shares the large files (`transcripts.zarr.zip` 3.5 GB, `morphology_focus/` 1.9 GB, `aux_outputs/` 0.3 GB) via hardlinks. These are stored once regardless of how many methods are in the bundle. Each new method adds only ~100–300 MB.

### Coordinate system
All outputs are in **original Xenium coordinates** (not subset-relative). The `segger_to_xenium.py` and `bidcell_to_xenium.py` scripts add `x_offset_um` / `y_offset_um` (from `subset_offsets.json`) to convert from subset space to original Xenium space.

## Results

| Method | Cells | % Transcripts assigned |
|--------|-------|----------------------|
| BIDCell | 2,687 | 3.6% |
| Segger | 148,487 | 14.3% |
| (10x default) | ~17,000 | — |
