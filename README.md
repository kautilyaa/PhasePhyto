# PhasePhyto

**A Rigorous OOD Study and Practical Recipe for Botanical Image Classification**

PhasePhyto is a controlled empirical study of what does and does not transfer
out-of-distribution for botanical image classification, using PlantVillage
(lab, clean) -> PlantDoc (field, cluttered) as the benchmark. The repo
implements a multi-stream architecture that fuses zero-parameter
frequency-domain transformations (Image Phase Congruency, PC) with a ViT-B/16
backbone and a CLAHE illumination stream via cross-attention, and reports
per-lever deltas across seven orthogonal OOD-hardening techniques.

> **Project thesis pivoted on 2026-04-22** (v0.1.18), and the ablation
> table closed 2026-04-23. The original claim -- "physics-informed fusion
> yields OOD-invariant features and beats a ViT baseline" -- is **not
> supported** by the data. `full` and `backbone_only` land at identical
> target accuracy (0.5229) with a +0.009 F1 delta inside run-to-run noise,
> and `pc_only` scores target F1 = 0.08 (random = 0.04). The project now
> reports this as a negative result and offers the rest of the training
> stack as a reusable OOD recipe. See **[PAPER.md](PAPER.md)** for the
> draft manuscript, [RESULTS.md](RESULTS.md) for the full run log, and
> the `invariance--classifier-head gap` section of the paper for the
> named failure mode.

## Current results (PlantVillage -> PlantDoc, 25-class mapped)

| Date | Run | Source Acc | Source F1 | Target Acc | Target F1 | Notes |
|---|---|---:|---:|---:|---:|---|
| 2026-04-17 | baseline (pre-OOD) | 99.72% | 0.9948 | 47.71% | 0.3603 | plain recipe |
| 2026-04-20 | `full` | 99.70% | 0.9951 | 50.33% | 0.3868 | +2.6 acc |
| 2026-04-21 | `pc_only` | 73.40% | 0.5878 | 9.15% | 0.0791 | PC alone fails OOD |
| 2026-04-21 | `full+leafmask` | 99.75% | 0.9955 | 52.29% | 0.3944 | +2.0 acc |
| 2026-04-23 | `no_fusion` | 99.75% | 0.9955 | 49.02% | 0.3464 | cross-attn = +0.048 target F1 |
| 2026-04-23 | `backbone_only` | 99.75% | 0.9955 | 52.29% | 0.3859 | **matches `full` (-0.009 F1)** |
| pending | pseudo-label rerun | -- | -- | -- | -- | Phase 7.1.b, calibrated thr |

**Three-week aggregate target movement:** +4.6 acc, +3.4 F1. Source
saturates at 99.7% throughout; residual gap is distributional.

**Ablation table closed (2026-04-23).** `full` and `backbone_only` land at
identical target accuracy (0.5229) with only +0.009 target F1 advantage
for `full`, inside run-to-run noise. The PC stream + cross-attention does
not beat plain label-smoothed ViT under this OOD recipe. The reusable
contribution is the training recipe, not the physics-informed fusion.

### Strict 3-class apple-overlap (PV -> PD / PP2021), 2026-04-26

A second benchmark on the **shared apple labels** (`Apple___healthy`,
`Apple___Apple_scab`, `Apple___Cedar_apple_rust`) across PV, PlantDoc, and
Plant Pathology 2021. Baseline ViT-B/16 trained on PV only (single seed):

| Target | n | Source Acc | Target Acc | Target F1 | Acc drop | F1 drop |
|---|---:|---:|---:|---:|---:|---:|
| PlantDoc test | 29 | 99.96% | 86.21% | 0.8632 | -13.8 pp | -13.6 pp |
| Plant Pathology 2021 | 11,310 | 99.96% | 71.36% | 0.6813 | -28.6 pp | -31.8 pp |

Per-class drops on PP2021 are asymmetric: `healthy` F1=0.80 (-20 pp),
`Apple_scab` 0.65 (-35 pp), `Cedar_apple_rust` 0.60 (-40 pp). The
confusion matrix shows two distinct failure modes -- (a) **healthy bias**
from PV's class imbalance (37% of actual scab and 9% of actual rust are
predicted as healthy), and (b) **rust collapses into scab** (43% of actual
rust). The first is a calibration problem; the second is a feature-shift
problem. See `Project_Summary.md` for the full per-class breakdown,
caveats, and suggested follow-ups.

This corroborates the negative-results headline at a tighter,
fully-overlapping label space (no label-mapping ambiguity), with a
real-world n=11,310 target. Caveat: single seed; PlantDoc-target n=29 is
statistically anecdotal.

## What this project does and does NOT claim

**Does claim:**

- A reproducible, documented training recipe for botanical OOD: label
  smoothing (0.1), differential LR (backbone 10x slower), weight EMA
  (decay 0.999), SAM (rho 0.05), strong augmentation with shared-mask
  random erasing and HSV-masked background replacement, HSV leaf
  foreground gate, hflip TTA, TENT test-time adaptation.
- A controlled ablation table (`full`, `pc_only`, `backbone_only`,
  `no_fusion`) attributing target-side gains to specific architectural
  components.
- A negative result: phase congruency features, despite being
  mathematically amplitude-invariant, do not transfer OOD on this leaf-disease
  benchmark. Classes that do transfer under `pc_only` share strong
  directional/periodic texture (leaf veins, lesion striations); hue- or
  context-defined classes collapse.

**Does NOT claim (explicitly rescinded 2026-04-22):**

- That physics-informed fusion beats a ViT baseline under OOD.
- That PC tokens contribute discriminative features (as opposed to
  regularization) to the fused representation.
- Any result on use cases 2-4 (histology, pollen, wood anatomy). The code
  supports those datasets but no experimental work is currently planned.

## Key mathematical property (holds at the raw-feature level only)

Phase congruency satisfies:

```
PC(image) = PC(image * k)    for any k > 0
```

This invariance is verified to within FP32 precision in cell 37 of the
training notebook across k in [0.5, 10]. But the `pc_only` ablation
demonstrates that the property does **not** propagate through the
classifier head to OOD class-discriminative power: the learned
classifier still over-fits source texture statistics even when its input
is amplitude-invariant. This gap between mathematical invariance and
learned-classifier invariance is itself one of the findings of the
negative study.

## Implementation state

- `make verify` passes locally on Python 3.11.5 (ruff, mypy, pytest).
- Core phase-congruency behaviour covered by tests for positive
  brightness-scaling invariance, orientation filter selectivity,
  circle-boundary phase symmetry, step-edge localisation, output ranges,
  and NaN safety.
- Full model forward/backward, attention normalisation, fusion parameter
  budget, source-only validation split handling, tiny synthetic CPU
  training, and semantic-only baseline output contract are tested.
- Colab notebook, setup guide, configs, Makefile targets, PhasePhyto
  CLIs, and baseline CLIs are present.
- GitHub Actions CI is present.

## Remaining work to close out the study (ROADMAP Phase 7.2)

1. ~~Finish `backbone_only` ablation~~ -- **done 2026-04-23** (target acc 52.29%, F1 0.3859). Matches `full` within run-to-run noise, settling the architectural question.
2. ~~Run `no_fusion` ablation~~ -- **done 2026-04-23** (target F1 = 0.3464; cross-attention contributes ~0.048 target F1 over mean-pool concat).
3. Rerun Phase 7.1.b pseudo-label with threshold ~0.7 (diagnostic on 2026-04-23 `no_fusion` run showed p95=0.904, so 0.9 admits only 19/153 samples, below the 50-sample floor; 0.7 would admit ~87). Run preferably on top of `backbone_only` given its flatter target trajectory.
4. One DANN (gradient reversal) attempt as the last target-gradient lever, only if pseudo-label at 0.7 fails to push target F1 past ~0.42.
5. Optional: save a "best-target-snapshot" checkpoint alongside "best-val" so the oracle target metric is always visible without dedicated ablations.
6. Write-up pass. The ablation table is complete and the empirical picture is closable.

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
>
> **Looking for the latest numbers?** See **[RESULTS.md](RESULTS.md)** for the chronological results log, including source/target metrics, deltas vs. prior runs, and per-run analysis.

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
   PhasePhyto-vs-baseline training/evaluation, with OOD-hardening enabled by
   default: strong augmentation + HSV-masked background replacement, label
   smoothing, differential backbone LR, weight EMA, an auxiliary PC-only
   classifier head, optional SAM (Foret et al. 2021), hflip TTA, TENT
   test-time adaptation (Wang et al. 2021), and an `ablation` toggle in
   `{"full","pc_only","backbone_only","no_fusion"}` that re-uses the same
   architecture (the forward path branches; parameter counts stay comparable).
   Artifact directories are auto-suffixed by ablation (e.g.
   `runs/20260420-HHMMSS_pc_only/`), so running all four ablations in sequence
   does not overwrite the previous run. See the `OOD-generalization knobs`
   block in CONFIG and the override cell below it.
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
PlantDoc class folders do not exactly match PlantVillage class names. The
notebooks and audit script now use a PlantDoc -> PlantVillage alias map so
PlantDoc classes such as `Apple Scab Leaf` evaluate against source classes such
as `Apple___Apple_scab` instead of producing an empty target set.

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

# Download Cassava Leaf Disease (Kaggle competition; requires accepting
# competition rules at https://www.kaggle.com/competitions/cassava-leaf-disease-classification/rules).
# Reorganizes Kaggle's flat train_images/ + train.csv into
# <root>/cassava/Cassava___<Disease>/<image>.jpg (PlantVillage-compatible
# ImageFolder layout, ~5.5 GB, 5 classes, ~21k images).
python scripts/download_data.py --dataset cassava --output data/plant_disease

# Download both
make data-all
```

### Additional datasets (staging / CLI-ready)

- **Cassava Leaf Disease** (Kaggle) is available end-to-end in the data
  pipeline: downloader CLI + notebook, `CassavaDataset`, registry support,
  batch-inference support, and config-driven train/eval entry points.
- **Plant Pathology 2021 / FGVC8** (Kaggle) is supported through the CLI and
  config stack. The downloader normalizes the competition CSV into
  ImageFolder layout using either the provided `labels` column or the
  one-hot disease columns, so the existing single-label pipeline can train on
  the observed label combinations. Use `configs/plant_pathology_2021.yaml`.
- **RoCoLe** (coffee leaf), **Rice Leaf Disease**, and **Banana Leaf Disease**
  are supported as ImageFolder-style benchmarks via the shared dataset
  registry, train/eval entry points, batch inference, and dedicated configs
  (`configs/rocole.yaml`, `configs/rice_leaf.yaml`,
  `configs/banana_leaf.yaml`). Their official hosts do not offer a stable
  programmatic API like Kaggle, so `scripts/download_data.py` expects
  `--source` pointing at a local extracted directory or archive downloaded
  from the official host, then normalizes the best raw/original ImageFolder
  candidate into the repo's standard layout.

### Strict apple-overlap benchmark

Use this when you want only the classes shared across PlantVillage,
PlantDoc, and Plant Pathology 2021:

- `Apple___healthy`
- `Apple___Apple_scab`
- `Apple___Cedar_apple_rust`

Prepare the overlap subset with:

```bash
python scripts/prepare_overlap_datasets.py \
  --plantvillage data/plant_disease/plantvillage \
  --plantdoc data/plant_disease/plantdoc \
  --plant-pathology-2021 data/plant_benchmarks/plant_pathology_2021 \
  --output data/overlap/apple_strict \
  --mode symlink
```

This writes:

```text
data/overlap/apple_strict/
  plantvillage/
  plantdoc/
  plant_pathology_2021/
  overlap_manifest.json
```

Overlap-ready configs:

- `configs/apple_overlap_plantdoc.yaml`
- `configs/apple_overlap_pp2021.yaml`

For a Colab-first, tar-backed workflow that downloads to Drive, archives the
raw datasets, builds/archives the overlap subset, hydrates to SSD, and then
trains/evaluates, use:

- `notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb`

Sample baseline numbers from this benchmark (single seed, PV-trained ViT-B/16,
2026-04-26): -13.8 pp accuracy on PlantDoc test (n=29), **-28.6 pp accuracy
and -31.8 pp F1 on Plant Pathology 2021 (n=11,310)**. Per-class breakdown
and failure-mode analysis are in `Project_Summary.md`.

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

### Numerical Stability: Split Epsilon in PhaseCongruencyExtractor

**What.** `PhaseCongruencyExtractor` uses two epsilon constants, not one:

- `self.eps = 1e-6` -- used **only** in the main PC denominators:
  `pc_orient = W_s * clamp(energy - T, 0) / (sum_A + self.eps)` and the
  equivalent phase-symmetry division.
- `self.sqrt_eps = 1e-12` -- used everywhere else: inside every
  `torch.sqrt(...)` call (amplitude, energy, oriented energy), inside the
  frequency-spread weight's `max_amp` denominator, and inside the per-image
  min-max normalizer `(hi - lo + sqrt_eps)`.

**Why.** A uniform `eps=1e-6` was tried first and broke amplitude invariance
at every `k`. The reason is that an additive constant in a denominator only
preserves `PC(kx) == PC(x)` when it is negligible compared to the denominator's
signal. `self.eps` sees two very different denominators:

1. `sum_A + eps` in the main PC formula. `sum_A` is a sum of 4 amplitudes per
   pixel, typically well above `1e-6` for any non-background pixel, so
   `eps=1e-6` does not perturb the ratio. This is where CLAUDE.md's
   "PC denominator eps >= 1e-6" rule is needed, because the gradient path
   through this division during training can explode if `sum_A` approaches
   zero.
2. `max_amp + eps` in the frequency-spread weight and `hi - lo + eps` in the
   min-max normalizer. Both of these denominators can be small on smooth or
   near-constant regions. A `1e-6` floor there shifts the ratio
   non-proportionally with `k`, which min-max normalization then amplifies
   across the whole `[0, 1]` output.

Splitting the epsilon keeps CLAUDE.md's NaN-gradient guarantee where it
matters (the PC denominator gradient path) and restores strict amplitude
invariance everywhere else.

**How the invariance test verifies this.** `notebooks/PhasePhyto_Colab.ipynb`
cell 37 runs `PC(test_img)` vs `PC(k * test_img)` for
`k in {0.1, 0.5, 2.0, 5.0, 10.0}` and compares the three PC maps elementwise.

- Test input is seeded (`torch.manual_seed(0)`) and uses a structured image
  (sine + cosine stripes + light noise) instead of pure uniform noise. Uniform
  noise has near-flat PC maps, which makes min-max normalization
  ill-conditioned and the test dominated by FP32 noise on near-tied maxima.
- Tolerance is tiered: `0.005` in the realistic illumination range
  `k in [0.5, 2.0]`, and `0.01`-`0.025` at extreme `k` where the hard
  `clamp(energy - T, 0)` threshold and FP32 precision interact at the
  noise-threshold boundary. At `k = 0.1` the signal shrinks 10x so FP32
  relative error grows 10x; the residual drift is cosmetic, not a model bug.

If you ever need bit-exact invariance at extreme `k`, cast the PC extractor to
FP64. This is not worth the speed hit during training, but is the only real
fix for extreme-k FP32 drift.

## Training Pipeline (OOD-hardened, PlantVillage -> PlantDoc)

Every design choice below targets one named failure mode from the prior
baseline run (99.72% source / 47.71% target, 52-point gap). The pipeline is
organized into seven phases in `notebooks/PhasePhyto_Colab.ipynb`.

### Phase 1 -- Data Prep (cells 19, 21)

**What.** `DualTransform` returns `(rgb_tensor, clahe_tensor, label)` per
sample: one RGB view for the ViT stream, one CLAHE-normalized view for the
illumination stream. A PlantDoc -> PlantVillage class alias map bridges the
two label namespaces (e.g. `Apple Scab Leaf` -> `Apple___Apple_scab`).

**Why.** The two streams need consistent but differently-preprocessed views of
the same image. PlantDoc folder names do not match PlantVillage class names,
so without the alias map the target set is empty, which is what the audit
script revealed in the first place.

**How.** `phasephyto/data/class_mapping.py` defines the alias dict. Cell 21
resolves the PlantDoc root across `test/Test/val/valid/base`, normalizes class
names (case/punct/underscore insensitive), and prints diagnostics (mapped
rows, skipped rows with reason, unknown folders, source classes with zero
target coverage).

### Phase 2 -- Strong Paired Augmentation (cell 19)

**What.** Applied in PIL space **before** CLAHE so both streams see the same
altered image:

- RandAugment (Cubuk et al. 2020)
- RandomPerspective, GaussianBlur, RandomRotation, stronger ColorJitter
- Shared-mask RandomErasing (same occlusion on RGB and CLAHE)
- HSV-saturation-masked background replacement (50% prob): leaf vs background
  detected by saturation threshold; non-leaf pixels swapped with one of three
  random styles -- solid noise, gradient, or high-frequency texture.

**Why.** PlantVillage has sterile white/green backgrounds. The baseline
learned the shortcut "uniform-green background -> healthy leaf" and collapsed
on PlantDoc's cluttered field backgrounds (hands, soil, bark, sky).
Background replacement simulates that shift during training so the classifier
cannot rely on background as a feature.

**How.** Saturation threshold `S > bg_saturation_thresh` masks leaf pixels;
non-leaf pixels are replaced with a random style. Both streams see the
replaced image downstream, so features learned are background-agnostic by
construction.

### Phase 3 -- Three Streams + Fusion + Ablation (cells 25, 26, 28, 30, 32, 34)

**What.**

- **PC stream** (cells 25-26): Log-Gabor filters -> `PhaseCongruencyExtractor`
  -> PC magnitude / phase symmetry / oriented energy maps -> shallow encoder
  -> 7x7 structural tokens.
- **ViT-B/16 stream** (cell 28): 86M pretrained backbone -> 196 patch tokens
  -> linear projection.
- **CLAHE stream** (cell 30): illumination-normalized CNN -> auxiliary
  semantic vector.
- **Cross-attention fusion** (cell 32): PC tokens = Q, semantic tokens = K/V.
- **Ablation toggle** (cell 34): `CONFIG["ablation"]` in
  `{"full", "pc_only", "backbone_only", "no_fusion"}`. Architecture stays
  identical across modes; only the forward path branches, so parameter counts
  are comparable.

**Why.**

- PC is *mathematically* amplitude-invariant: `PC(kx) = PC(x)`. Lighting
  changes destroy pixel gradients but preserve phase structure. This is the
  only stream that is physics-guaranteed to survive lab-to-field lighting.
- ViT gives strong semantic priors (ImageNet pretraining) but is pixel
  intensity dependent.
- Cross-attention lets the physics-invariant structural tokens **query** the
  semantic knowledge, rather than averaging streams or concatenating.
- The ablation toggle produces the evidence table that proves each stream is
  actually pulling its weight -- without it, we cannot claim fusion helps.

**How.** `PhasePhyto.forward(rgb, clahe)` runs all three streams, fuses, and
classifies. The ablation branch only decides which pooled tokens feed the
classification head; the physical modules are unchanged, so PC maps,
structural tokens, and ViT features are always computed.

### Phase 4 -- Loss and Optimization (cells 39, 41)

**What.**

- **Label-smoothed cross-entropy** (eps=0.1) replaces FocalLoss. FocalLoss is
  still defined for reproducibility of earlier runs but is unused.
- **Differential LR:** ViT backbone runs at 10x lower LR than PC / fusion /
  head via `_build_param_groups`. Warmup is per-group-aware.
- **Weight EMA** (decay 0.999): a shadow copy updated every step. Validation
  and checkpointing use EMA weights; final EMA is always saved separately.
- **Auxiliary PC-only head** (`aux_pc_head` in cell 34): a small classifier
  on the mean-pooled structural tokens only. Training loss adds
  `aux_pc_weight * CE(aux_logits, labels)`.
- **SAM (Foret et al. 2021)** optional: two-step forward/backward that
  minimizes the sharpness of the loss surface. AMP is disabled when SAM is
  active (the double forward breaks AMP's backward graph).

**Why.**

- Label smoothing prevents over-confident source logits, which hurt OOD
  calibration. FocalLoss on a near-saturated source set made this worse.
- Backbone domination: 86M ViT parameters vs. ~150K PC parameters. Without a
  lower backbone LR, ViT gradients overwhelm PC gradients and the model
  degenerates into a regularized ViT that cannot generalize OOD.
- EMA smooths the weight trajectory. The prior baseline's best val F1 fired
  at epoch 1 on a near-saturated source -- EMA weights after a few epochs are
  a better OOD generalizer than that early-epoch best-val point.
- The auxiliary PC head forces the PC stream alone to be class-discriminative,
  preventing the ViT stream from nullifying PC gradients just because ViT
  alone already fits the source.
- SAM finds flatter minima, which empirically transfer better to shifted
  distributions.

**How.** `train_epoch` (AMP path) and `train_epoch_sam` (no AMP) are separate
functions. `ModelEMA` maintains shadow parameters. The training loop branches
on `CONFIG["use_sam"]` and writes two checkpoints: `best_phasephyto.pt` (best
val F1) and `final_ema_phasephyto.pt` (end-of-training EMA).

### Phase 5 -- Test-Time Adaptation (cell 45)

**What.**

- **TENT (Wang et al. 2021):** deep-copy the trained model, freeze all
  parameters except BatchNorm / LayerNorm affines (weight + bias), set BN to
  `train()` so running stats update, minimize Shannon entropy of target
  predictions for `tent_steps` steps at `tent_lr`.
- **hflip TTA:** at evaluation, softmax-average the image and its horizontal
  flip.

**Why.**

- TENT adapts a trained model to the target distribution without target
  labels, using only the observation that a well-calibrated model should be
  confident on the correct class. BN/LN affines have enough degrees of
  freedom to re-align feature distributions without catastrophic forgetting.
- hflip TTA is a near-free accuracy gain (plant images are left-right
  symmetric).
- Source evaluation runs on the untouched model copy so source metrics stay
  honest and do not reflect target-side adaptation.

**How.** `tent_adapt(model, target_loader, steps, lr)` returns an adapted
copy. Cell 45 loads best or final-EMA weights (auto-prefers final-EMA when
best was saved in `epoch < 2`, which indicates overfitting to source rather
than learning), runs TENT on a copy for target eval, and evaluates the
untouched model on source.

### Phase 6 -- Sanity and Invariance Verification (cells 36, 37)

**What.**

- Cell 36 forces `ablation="full"` and asserts PC map shapes and
  forward/backward success.
- Cell 37 verifies `PC(k * image) == PC(image)` for
  `k in {0.1, 0.5, 2, 5, 10}` with tiered tolerance.

**Why.** Amplitude invariance is the *physical claim* the architecture rests
on. If cell 37 fails the PC stream is not contributing scale-invariant
features, and the design collapses to a regularized ViT. Worth running before
every training run.

**How.** Seeded structured test image (sine + cosine stripes + light noise),
tiered tolerance by realistic illumination range -- see "Numerical Stability"
above. `self.eps=1e-6` in the PC denominator (NaN-gradient safety per
CLAUDE.md); `self.sqrt_eps=1e-12` everywhere else (otherwise eps pollutes
small denominators and breaks invariance).

### Phase 7 -- Ablation and Post-Run Inspection (notebooks 00-04)

**What.** Run the training notebook four times with `CONFIG["ablation"]` set
to each of `{"full", "pc_only", "backbone_only", "no_fusion"}`. Each run
writes to its own `runs/<timestamp>_<ablation>/` directory (checkpoints,
metrics, plots, manifest). The `PhasePhyto_Inspect_*` notebooks tabulate them.

**Why.** This is the evidence that the fusion is doing useful work. If
`full` ~= `backbone_only`, the PC stream is not helping. If `full` >
`no_fusion`, cross-attention is genuinely better than simple averaging. Any
gap-closure claim requires this table.

**How.** `RUN_NAME` auto-suffixes with the ablation key so sequential runs do
not clobber each other; the inspector notebooks read the manifest across run
directories and produce the comparison table.

### Phase 7.1.a -- OOD Foreground Segmentation & Diagnostic Hooks (v0.1.15)

**Trigger.** 2026-04-21 `pc_only` ablation (target F1 = 0.08, see
`RESULTS.md`) showed the PC stream's learned features do not transfer to
PlantDoc. The most likely cause: PC is computing phase structure over
PlantVillage studio backgrounds, which do not exist on PlantDoc.

**What.** Five CONFIG knobs in the training notebook, all default off:

| Knob | Type | Purpose |
|------|------|---------|
| `leaf_mask_mode` | `"off" | "hsv" | "hsv_blur"` | Gate PC to leaf pixels. `"hsv"` flattens non-leaf pixels to the leaf-mean colour; `"hsv_blur"` replaces them with a gaussian-blurred version of the image. |
| `leaf_mask_sat_thresh` | int (default 40) | Min HSV saturation to count as foreground. |
| `leaf_mask_blur_sigma` | float (default 1.5) | Gaussian sigma when `mode == "hsv_blur"`. |
| `checkpoint_every` | int (default 0) | If `> 0`, save a checkpoint every N epochs to `runs/<ts>/checkpoints/epoch_<N>.pt`. |
| `target_snapshot_every` | int (default 0) | If `> 0`, run target eval every N epochs during training and record into `history["target_snapshot_*"]`. |

**Why.** The leaf mask is the highest-expected-impact lever given the
`pc_only` finding: a foreground-gated PC stream has no access to
background phase structure in the first place, so it cannot learn to
shortcut on it. The diagnostic hooks exist because source val F1
saturates early on this recipe (`backbone_only` hit 0.9917 at epoch 3);
the "best" checkpoint is a poor proxy for target transfer, so we want to
see target trajectory directly and to keep multiple checkpoints for
post-hoc selection.

**How.** Applied inside `DualTransform` so the mask gates pixels BEFORE
both CLAHE and PC computation, and applied to both train and val
transforms so target (PlantDoc) eval inherits the gating. Planned first
runs are logged in the `RESULTS.md` pending-runs table:
`<ts>_full_leafmask` (`"hsv"`) and `<ts>_full_leafmask_blur`
(`"hsv_blur"`), both with `target_snapshot_every=3` and
`checkpoint_every=3`.

### Phase 7.1.b -- Pseudo-Label Self-Training (v0.1.16)

**Trigger.** Both `full` and `backbone_only` saturate source validation
at >= 0.99. When source is this saturated, more source-side
regularisation has no effect -- only target-side gradient can close the
remaining gap. Pseudo-labeling is the smallest lever that adds that
signal.

**What.** After the main training loop, an optional self-training phase
(cell 43 in the patched notebook):

1. Uses the EMA model (if enabled) to predict on target with no grad.
2. Keeps samples where `max(softmax) >= pseudo_label_threshold`.
3. Treats those predictions as ground truth, builds a target dataset
   with **train-time augmentation** applied.
4. Fine-tunes on `ConcatDataset(source_subset, pseudo_target)` for
   `pseudo_label_epochs` epochs at `lr * pseudo_label_lr_mult` with a
   fresh AdamW optimiser (no SAM -- two-step + noisy labels interacts
   poorly).
5. Saves to `pseudo_phasephyto.pt`; the evaluation cell (cell 47)
   prefers it over best-val / final-EMA checkpoints.

**Knobs.**

| Knob | Default | Purpose |
|------|---------|---------|
| `use_pseudo_label` | `True` (as of v0.1.16) | Master toggle. |
| `pseudo_label_threshold` | `0.9` | Higher -> fewer, cleaner labels. |
| `pseudo_label_epochs` | `5` | Short fine-tune -- more invites drift. |
| `pseudo_label_lr_mult` | `0.1` | Of `CONFIG["lr"]`. Low to avoid forgetting. |
| `pseudo_label_min_samples` | `50` | Skip phase if fewer confident samples. |

**Why it should help here specifically.** PlantDoc class distribution
is such that most images are leaves similar to something in
PlantVillage in colour and shape (that is why the class alias map
works at all). Even a mediocre model produces confidently-correct
predictions for the easier target classes. Those labeled anchors let
us fine-tune over the harder classes via shared representations. The
high confidence threshold plus keeping source in the loss bounds
confirmation bias.

**What's held as last resort.** DANN with gradient reversal on the
fused features is architecturally larger (adds a domain discriminator
branch) and empirically inconsistent on shifts of this kind. Kept in
ROADMAP Phase 7.1.c for the case where the combined leaf-mask +
pseudo-label run does not close the gap below 10 points.

**Safety nets added in v0.1.17.**

- Pseudo-label phase prints a target max-softmax decile histogram and
  counts above common thresholds *before* filtering, so the threshold
  can be tuned once per dataset rather than by rerunning.
- `aux_pc_weight` is hard-zeroed during the pseudo-label fine-tune,
  since `pc_only` already established the PC stream is decorative on
  OOD and its aux head only wastes fine-tune budget.
- Full `history` dict (including target snapshots) is written to
  `RESULTS_DIR/history.json` at end of training so the OOD trajectory
  survives a kernel restart after training but before eval.
- Every `torch.save` call embeds the full CONFIG dict, so a checkpoint
  can be re-attributed to its ablation + knob set months later.
- A new leaf-mask sanity-viz cell (index 52) renders source + target
  samples alongside their HSV foreground masks with per-image
  foreground fraction. Run it once before committing to a 15-epoch run
  to verify the threshold works for both datasets.

### How the phases compose

Each phase targets one failure mode of the prior 47.71% baseline:

| Failure mode | Addressed by |
|---|---|
| Background-as-shortcut | Phase 2 (bg replace), Phase 3 (PC stream is background-agnostic by construction) |
| ViT dominates PC | Phase 4 (differential LR, auxiliary PC head) |
| Source over-confidence | Phase 4 (label smoothing, EMA, SAM) |
| Target distribution drift at eval | Phase 5 (TENT on BN/LN affines, hflip TTA) |
| Architectural claim unverified | Phase 6 (invariance test), Phase 7 (ablation) |
| Label-namespace mismatch | Phase 1 (class alias map + audit) |
| FP32 / eps interaction | Numerical Stability section above (split epsilon) |
| PC fits source background | Phase 7.1.a (HSV leaf foreground mask) |
| Best-val checkpoint != best-target | Phase 7.1.a (periodic checkpoint + target snapshot) |
| Source saturated, no target signal | Phase 7.1.b (pseudo-label self-training on high-confidence target) |

The design is not "try everything and hope" -- each lever maps to a named
failure mode observed in the prior run, with a mechanism that explains why
the lever should help **OOD specifically** rather than just boost source
accuracy.

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
