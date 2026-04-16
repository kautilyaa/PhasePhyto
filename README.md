# PhasePhyto

**Physics-Informed Differentiable Phase Congruency for Cross-Domain Botanical Visual Recognition**

PhasePhyto is a three-stream hybrid architecture that fuses zero-parameter frequency-domain transformations (Image Phase Congruency) with parameterised semantic neural networks for domain-invariant botanical image classification.

## Overall Project State

PhasePhyto is currently a **verified implementation scaffold with baseline
tooling, ready for the first real benchmark run**.

What is verified:

- `make verify` passes locally on Python 3.11.5.
- Ruff passes, mypy reports no source issues, and pytest reports 30 passing
  tests.
- Core phase-congruency behavior is covered by tests for positive
  brightness-scaling invariance, orientation filter selectivity, circle-boundary
  phase symmetry, step-edge localization, output ranges, and NaN safety.
- Full model forward/backward, attention normalization, fusion parameter budget,
  source-only validation split handling, tiny synthetic CPU training, and
  semantic-only baseline output contract are tested.
- Colab training notebook, setup guide, configs, Makefile targets, PhasePhyto
  CLIs, and baseline CLIs are present.
- Colab notebook requirements and embedded training code are aligned with the
  verified repo implementation: `torch.amp`, source-only validation,
  PlantDoc `test/` split resolution, and corrected Log-Gabor FFT grid/sign masks.
- The import package is now `phasephyto`, Git has been initialized locally, and
  GitHub Actions CI is present.

What is still pending:

- Real PlantVillage -> PlantDoc PhasePhyto-vs-baseline numbers are not recorded
  yet.
- PlantVillage/PlantDoc class overlap should be audited with
  `scripts/audit_class_overlap.py` on the downloaded real data before publishing
  benchmark claims.
- Run `scripts/benchmark.py` to produce the first reproducible benchmark table.

## Key Innovation

Standard deep learning models suffer 15--37% accuracy drops when moving from lab to field conditions because they rely on pixel intensity gradients that are destroyed by shadows, lighting changes, and sensor noise.

PhasePhyto exploits a mathematical property of phase congruency:

```
PC(image) = PC(image * k)    for any k > 0
```

This makes the phase-congruency stream invariant to positive brightness scaling
in unit tests, so structural features (edges, boundaries, textures) are preserved
under illumination changes. End-to-end domain-shift gains still need to be
measured against the included semantic-only baseline on your real source/target
data.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              Input RGB Image              │
                    └──────┬───────────────┬───────────────┬───┘
                           │               │               │
                    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                    │  Grayscale  │ │   Raw RGB   │ │ CIELAB+CLAHE│
                    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                           │               │               │
                    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                    │ FFT + 24    │ │  ViT-B/16   │ │ Shallow CNN │
                    │ Log-Gabor   │ │  Backbone   │ │  (2 layers) │
                    │ Filters     │ │             │ │             │
                    │ (0 params)  │ │ (86M params)│ │ (~50K)      │
                    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                           │               │               │
                    ┌──────▼──────┐ ┌──────▼──────┐        │
                    │ PC Maps:    │ │  196 Patch  │        │
                    │ • Magnitude │ │  Tokens     │        │
                    │ • Symmetry  │ │  (14x14)    │        │
                    │ • Orient.E  │ │             │        │
                    └──────┬──────┘ └──────┬──────┘        │
                           │               │               │
                    ┌──────▼──────┐        │               │
                    │ PC Encoder  │        │               │
                    │ 49 Struct.  │        │               │
                    │ Tokens(7x7) │        │               │
                    └──────┬──────┘        │               │
                           │               │               │
                      Q ───┘         K,V ──┘               │
                           │               │               │
                    ┌──────▼───────────────▼──────┐        │
                    │    Cross-Attention Fusion    │        │
                    │         (~331K params)       │        │
                    └──────────────┬───────────────┘        │
                                  │                        │
                           ┌──────▼────────────────────────▼──┐
                           │  Concatenate + MLP Classifier     │
                           └──────────────┬────────────────────┘
                                          │
                                   ┌──────▼──────┐
                                   │   Logits    │
                                   └─────────────┘
```

## Use Cases

| # | Domain | Source | Target | Gap Addressed |
|---|--------|--------|--------|---------------|
| 1 | Plant Disease | PlantVillage (lab) | PlantDoc (field) | Lighting, shadows, sensor noise |
| 2 | Plant Histology | Single stain | Cross-stain | Safranin-O / Toluidine Blue / Lugol's |
| 3 | Palynology | Curated microscopy | Multi-microscope | Illumination, depth-of-field |
| 4 | Wood Anatomy | XyloTron lab | Field specimens | Moisture, sanding, ambient light |

## Quick Start

> **New to PhasePhyto?** See **[GUIDE.md](GUIDE.md)** for a complete step-by-step walkthrough covering Colab setup, data download, training, evaluation, inference, reading PC maps, and troubleshooting.

### Installation

```bash
# Clone and install
git clone https://github.com/<your-username>/PhasePhyto.git
cd PhasePhyto
pip install -e ".[all]"
```

### Google Colab (Recommended for first run)

The Colab workflow is split by purpose:

1. `notebooks/PhasePhyto_Download_Data_To_Drive.ipynb` -- one-time data download
   and Drive manifest creation.
2. `notebooks/PhasePhyto_Colab.ipynb` -- synthetic smoke test and full
   PhasePhyto-vs-baseline training/evaluation.
3. `notebooks/PhasePhyto_Inspect_00_Index.ipynb` -- start post-run inspection.
   Then use the focused inspector notebooks for run overview, metrics, plots,
   or reports.

Open `notebooks/PhasePhyto_Colab.ipynb` in Google Colab with a T4 GPU. The
notebook is fully self-contained -- it installs dependencies (`timm`,
`opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`,
`pyyaml`), implements the architecture, trains on synthetic data by default
(`USE_SYNTHETIC = True`) or PlantVillage/PlantDoc with real paths, evaluates
domain shift, trains the ViT baseline, and visualises PC maps. The notebook
uses source-only validation and resolves nested PlantDoc layouts such as
`plantdoc/test/<class>`.

The notebook also separates pipeline storage from data import. Set
`CONFIG["storage_backend"]` to:

- `"drive"` to write all run artifacts to Google Drive,
- `"colab_ssd"` to write to fast ephemeral Colab local SSD,
- `"external_ssd"` to write to a mounted/hooked SSD path such as
  `CONFIG["external_ssd_project_dir"]`.

Each run gets a structured directory:
`runs/<run_name>/{checkpoints,plots,results}` plus `run_manifest.json`.
For real data, prefer Drive tar archives: the downloader notebook creates
`plantvillage.tar` and `plantdoc.tar`, and the training notebook extracts them
into `/content/data` before training. This is much faster than `rsync` or
reading thousands of image files directly from Drive.
Notebook quality checks currently cover JSON validity, Python code-cell syntax,
Drive/SSD artifact paths, label-aligned PlantDoc classification reports and
confusion matrices, and reusable run manifests.
If Colab encounters `PIL.UnidentifiedImageError`, see `notebooks/README.md`
for the corrupt-image scan/quarantine workflow before rerunning training.

### Data Download

For Colab-first workflows, use
`notebooks/PhasePhyto_Download_Data_To_Drive.ipynb` once to download
PlantVillage and PlantDoc into Google Drive. It writes:

```text
/content/drive/MyDrive/PhasePhyto/data/plant_disease/
  plantvillage/
  plantdoc/
  dataset_manifest.json
/content/drive/MyDrive/PhasePhyto/data/archives/
  plantvillage.tar
  plantdoc.tar
```

Then use `PhasePhyto_Colab.ipynb` with archive hydration enabled:

```python
CONFIG["use_synthetic"] = False
CONFIG["hydrate_local_data_from_archives"] = True
CONFIG["drive_archive_dir"] = "/content/drive/MyDrive/PhasePhyto/data/archives"
```

```bash
# Generate synthetic data for pipeline testing (no download needed)
make data-synthetic
# -- or --
python scripts/download_data.py --dataset synthetic --output data/synthetic

# Download PlantVillage from Kaggle (requires ~/.kaggle/kaggle.json)
make data-plantvillage
# -- or --
python scripts/download_data.py --dataset plantvillage --output data/plant_disease

# Download PlantDoc from GitHub
make data-plantdoc
# -- or --
python scripts/download_data.py --dataset plantdoc --output data/plant_disease

# Download both
make data-all
```

### Makefile Targets

```bash
make help            # Show all available targets
make install         # pip install -e ".[all]"
make verify          # Run lint + typecheck + tests (all-in-one)
make lint            # ruff check phasephyto/ tests/ scripts/
make lint-fix        # Auto-fix lint issues
make typecheck       # mypy phasephyto/
make test            # pytest tests/ -v
make test-pc         # Phase congruency tests only
make test-model      # Model forward/backward tests only
make train CONFIG=configs/plant_disease.yaml
make train-baseline CONFIG=configs/plant_disease.yaml
make evaluate CONFIG=configs/plant_disease.yaml CKPT=checkpoints/best_model.pt
make evaluate-baseline CONFIG=configs/plant_disease.yaml CKPT=checkpoints/plant_disease/baseline/best_model.pt
make audit-classes SOURCE=data/plant_disease/plantvillage TARGET=data/plant_disease/plantdoc
make benchmark CONFIG=configs/plant_disease.yaml
make clean           # Remove generated/cache files
```

### Current Verification Status

The repository currently verifies with:

```bash
make verify
```

Expected result: Ruff passes, mypy reports no source issues, and pytest reports
30 passing tests. The test suite covers phase-congruency brightness invariance,
orientation selectivity, circle-boundary phase symmetry, step-edge localization,
model forward/backward behavior, attention normalization, source-only validation
split handling, fusion parameter budget, a tiny synthetic CPU training smoke
test, and the semantic-only baseline output contract.

### Benchmark Protocol

Before making PlantVillage -> PlantDoc claims, audit class overlap and run both
models on the same split:

```bash
python scripts/audit_class_overlap.py \
    --source data/plant_disease/plantvillage \
    --target data/plant_disease/plantdoc \
    --output benchmark_results/class_overlap.json \
    --fail-on-empty

python scripts/benchmark.py \
    --config configs/plant_disease.yaml \
    --output-dir benchmark_results
```

Training now validates on source-domain data only. If `data.val_dir` is unset
and no source validation folder exists, `phasephyto.train` creates a deterministic
source-domain random split using `data.val_split`. The target domain is reserved
for final OOD evaluation.

### Training

```bash
# Train on plant disease (PlantVillage -> PlantDoc)
python -m phasephyto.train --config configs/plant_disease.yaml

# Train on cross-stain histology
python -m phasephyto.train --config configs/histology.yaml

# Override config from command line
python -m phasephyto.train --config configs/base.yaml --override training.lr=1e-4 training.epochs=30

# Train a semantic-only timm baseline for comparison
python -m phasephyto.train_baseline --config configs/plant_disease.yaml
```

### Evaluation

```bash
# Evaluate domain shift (supports all 4 use cases)
python -m phasephyto.evaluate \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/plant_disease/best_model.pt \
    --source-dir data/plant_disease/plantvillage \
    --target-dir data/plant_disease/plantdoc/test

# Cross-stain histology evaluation
python -m phasephyto.evaluate \
    --config configs/histology.yaml \
    --checkpoint checkpoints/histology/best_model.pt \
    --source-stain safranin --target-stain toluidine

# Evaluate the semantic-only baseline on the same source/target split
python -m phasephyto.evaluate_baseline \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/plant_disease/baseline/best_model.pt \
    --source-dir data/plant_disease/plantvillage \
    --target-dir data/plant_disease/plantdoc/test
```

### Inference with PC Map Visualisation

```bash
# Single image with Grad-CAM
python -m phasephyto.inference \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/best_model.pt \
    --input path/to/image.jpg \
    --gradcam

# Batch inference on a directory
python -m phasephyto.inference \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/best_model.pt \
    --input path/to/image_dir/
```

## Important: TransformSubset Pattern

When using `random_split` to create train/val splits, the resulting `Subset` objects inherit the parent dataset's transform. This means **val data gets training augmentations** (random crops, flips, color jitter), silently inflating validation metrics.

Always wrap val subsets with `TransformSubset`:

```python
from phasephyto.data.datasets import TransformSubset
from torch.utils.data import random_split

train_split, val_split = random_split(dataset, [train_size, val_size])
val_subset = TransformSubset(val_split, val_transforms)  # override with deterministic transforms
```

## Project Structure

```
PhasePhyto/
├── .github/workflows/ci.yml      # CI: install + make verify
├── CLAUDE.md                    # Agent directives + project context
├── ROADMAP.md                   # Implementation roadmap (10 phases)
├── README.md                    # This file
├── Makefile                     # Verification, train/eval, audit, benchmark targets
├── pyproject.toml               # Dependencies and build config
├── configs/
│   ├── base.yaml                # Shared defaults
│   ├── plant_disease.yaml       # PlantVillage -> PlantDoc
│   ├── histology.yaml           # Cross-stain potato tuber
│   ├── pollen.yaml              # Pollen grain classification
│   └── wood.yaml                # XyloTron wood anatomy
├── phasephyto/
│   ├── models/
│   │   ├── phase_congruency.py  # Log-Gabor filter bank + PC extraction
│   │   ├── pc_encoder.py        # PC maps -> Structural Tokens
│   │   ├── semantic_backbone.py # ViT/CNN backbone wrapper (timm)
│   │   ├── illumination_norm.py # CIELAB + CLAHE + shallow CNN
│   │   ├── cross_attention.py   # Structural-semantic fusion
│   │   └── phasephyto.py        # Full model assembly
│   ├── data/
│   │   ├── datasets.py          # Dataset classes (4 use cases) + TransformSubset
│   │   ├── splits.py            # Split resolution + class-count helpers
│   │   └── transforms.py        # DualTransform (RGB+CLAHE), augmentation pipeline
│   ├── training/
│   │   ├── trainer.py           # Training loop (AMP, grad clip, early stopping)
│   │   └── losses.py            # Focal loss, label smoothing CE
│   ├── evaluation/
│   │   ├── metrics.py           # Accuracy, F1, confusion matrix
│   │   ├── domain_shift.py      # Source vs target evaluation protocol
│   │   └── xai.py               # Grad-CAM + attention visualisation
│   ├── utils/
│   │   ├── config.py            # YAML config with dataclass validation
│   │   └── seed.py              # Reproducibility
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point (all 4 use cases)
│   └── inference.py             # Inference + PC visualisation + Grad-CAM
├── tests/
│   ├── test_phase_congruency.py # Filter bank + PC map correctness + amplitude invariance
│   ├── test_data_protocol.py    # Source-only validation + split resolution
│   └── test_model_forward.py    # End-to-end forward/backward + shapes + param counts
├── notebooks/
│   ├── README.md                # Notebook workflow index
│   ├── PhasePhyto_Download_Data_To_Drive.ipynb # One-time Drive dataset prep
│   ├── PhasePhyto_Colab.ipynb   # Training/evaluation notebook (48 cells)
│   ├── PhasePhyto_Inspect_00_Index.ipynb # Inspection index/run chooser
│   ├── PhasePhyto_Inspect_01_Run_Overview.ipynb # Manifest/artifact viewer
│   ├── PhasePhyto_Inspect_02_Metrics.ipynb # Metrics comparison
│   ├── PhasePhyto_Inspect_03_Plots.ipynb # Plot/image viewer
│   ├── PhasePhyto_Inspect_04_Reports.ipynb # Report viewer
│   └── PhasePhyto_Inspect_Run_Results.ipynb # Compatibility pointer
└── scripts/
    ├── audit_class_overlap.py   # Source/target class-overlap audit
    ├── benchmark.py             # PhasePhyto-vs-baseline benchmark orchestration
    └── download_data.py         # Download PlantVillage, PlantDoc, or generate synthetic
```

## Phase Congruency Maps

The PC stream produces three complementary structural maps from each input image:

| Map | Detects | Botanical Use |
|-----|---------|---------------|
| **PC Magnitude** | Scale-invariant edges and boundaries | Lesion edges, leaf margins, cell walls |
| **Phase Symmetry** | Symmetric morphological structures | Spore bodies, stomata, starch granules, pollen pores |
| **Oriented Energy** | Directional texture patterns | Vein networks, fungal hyphae, vascular bundles, wood rays |

All three maps are tested for positive brightness-scaling invariance. Treat
claims about real lab-to-field improvement as experimental until PhasePhyto and
the baseline have both been trained and evaluated on the same split.

## Configuration

All configs inherit from `configs/base.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone_name` | `vit_base_patch16_224` | timm backbone (ViT or CNN) |
| `model.fusion_dim` | 256 | Cross-attention embedding dimension |
| `model.pc_scales` | 4 | Log-Gabor wavelet scales |
| `model.pc_orientations` | 6 | Log-Gabor filter orientations |
| `training.lr` | 1e-4 | Base learning rate (AdamW) |
| `training.loss` | `cross_entropy` | Loss: `cross_entropy`, `focal`, `label_smoothing` |
| `training.patience` | 10 | Early stopping patience |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test PC invariance specifically
python -m pytest tests/test_phase_congruency.py -v -k "invariance"

# Test full model forward/backward
python -m pytest tests/test_model_forward.py -v
```

## References

1. Kovesi, P. (1999). "Image Features from Phase Congruency." *Videre: Journal of Computer Vision Research*, 1(3).
2. Vidyarthi et al. (2024). "PhaseHisto: Differentiable Phase Congruency for Cross-Domain Histopathology."
3. Hughes, D.P. & Salathé, M. (2015). "An open access repository of images on plant health to enable the development of mobile disease diagnostics." *arXiv:1511.08060*.
4. Singh, D. et al. (2020). "PlantDoc: A Dataset for Visual Plant Disease Detection." *CODS-COMAD*.
5. Hermanson, J.C. & Wiedenhoeft, A.C. (2011). "A brief review of machine vision in the context of automated wood identification systems." *IAWA Journal*.
6. Dosovitskiy, A. et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
7. Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.

## License

Apache-2.0
