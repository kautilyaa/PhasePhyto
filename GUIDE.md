# PhasePhyto: Complete User Guide

## Table of Contents

1. [Option A: Run on Google Colab (Recommended)](#option-a-google-colab)
2. [Option B: Run Locally](#option-b-run-locally)
3. [Getting the Data](#getting-the-data)
4. [Training PhasePhyto](#training-phasephyto)
5. [Evaluating Domain Shift](#evaluating-domain-shift)
6. [Running Inference on Your Own Images](#running-inference-on-your-own-images)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)
9. [Workflow Cheat Sheet](#workflow-cheat-sheet)

---

## Current State

This project is in a **verified implementation / pre-real-benchmark** state.

- Local verification currently passes with `make verify`:
  - Ruff passes.
  - mypy reports no source issues.
  - pytest reports 30 passing tests.
- The Colab notebook, configs, setup guide, PhasePhyto training/evaluation CLIs,
  and semantic-only baseline CLIs are available.
- The Colab notebook is synced with the verified implementation: it installs the
  notebook-specific dependencies, uses source-only validation, resolves nested
  PlantDoc `test/` folders, and uses the corrected Log-Gabor FFT grid/sign-mask
  implementation.
- The local package imports as `phasephyto`, with CI and benchmark/audit scripts
  in place.
- The next scientific milestone is to run PhasePhyto and the baseline on the
  same real PlantVillage -> PlantDoc split, audit class overlap, and record the
  resulting source/target metrics.
- Until those real-data numbers are recorded, treat domain-shift improvement
  claims as hypotheses to validate rather than established results.

---

## Option A: Google Colab

This is the fastest way to get results. No local setup needed.

The notebook installs these requirements in Colab:

```bash
timm opencv-python-headless scikit-learn matplotlib seaborn tqdm pyyaml
```

PyTorch, torchvision, and Pillow are provided by the standard Colab GPU runtime.

### Step 1: Open the Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File > Upload notebook**
3. Upload `notebooks/PhasePhyto_Colab.ipynb` from this repo
4. Or, if you pushed to GitHub: **File > Open notebook > GitHub** and paste the repo URL

### Step 2: Select GPU Runtime

1. Click **Runtime > Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

### Step 3: Run with Synthetic Data (Pipeline Test)

Run every cell top to bottom. With `USE_SYNTHETIC = True` (the default), the notebook:
- Generates fake images with class-specific frequency patterns
- Trains PhasePhyto for 20 epochs (~5 min on T4)
- Evaluates domain shift (source vs target)
- Visualises PC maps and attention overlays
- Trains a baseline ViT for comparison
- Produces a comparison table

This verifies the full pipeline works before you invest time downloading real data.
The synthetic target split uses stronger brightness variation than the synthetic
source split, so the source-vs-target evaluation exercises the illumination-shift
pipeline without requiring real downloads.

### Step 4: Run with Real Data

Once the pipeline test passes:

1. Set `USE_SYNTHETIC = False` in the Configuration cell
2. Get PlantVillage data (choose one method):

**Method A -- Kaggle API (recommended):**
```python
# Upload your kaggle.json to Colab, then uncomment these lines in cell 8:
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d abdallahalidev/plantvillage-dataset -p /content/data
!unzip -q /content/data/plantvillage-dataset.zip -d /content/data/plantvillage_raw
```
To get kaggle.json: Go to kaggle.com > Your Profile > Account > Create New Token.

**Method B -- Google Drive:**
```python
# Download PlantVillage to your Google Drive first, then:
from google.colab import drive
drive.mount('/content/drive')
PLANTVILLAGE_DIR = Path('/content/drive/MyDrive/datasets/plantvillage')
PLANTDOC_DIR = Path('/content/drive/MyDrive/datasets/plantdoc')
```

3. Get PlantDoc data:
```bash
!git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git /content/data/plantdoc
```

4. Update the `PLANTVILLAGE_DIR` and `PLANTDOC_DIR` paths to point to your data.
   If PlantDoc is laid out as `plantdoc/test/<class>`, the notebook resolves the
   `test` folder automatically.
5. Run all cells again

The notebook trains/validates only on PlantVillage/source data. PlantDoc is used
only for final target-domain evaluation and baseline comparison.

### Expected Colab Outputs

| Output File | What It Contains |
|-------------|-----------------|
| `training_curves.png` | Loss, accuracy, F1 over epochs |
| `illumination_invariance.png` | PC maps at 5 brightness levels proving invariance |
| `analysis_sample_*.png` | Per-image: original, PC magnitude, phase symmetry, oriented energy, attention |
| `confusion_matrices.png` | Baseline ViT vs PhasePhyto on target domain |
| `best_phasephyto.pt` | Best model checkpoint |
| `baseline_vit.pt` | Baseline ViT checkpoint |
| `phasephyto_results.json` | All metrics in machine-readable format |

---

## Option B: Run Locally

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended, 8GB+ VRAM)
- ~5GB disk space for PlantVillage dataset

### Step 1: Install

```bash
git clone https://github.com/<your-username>/PhasePhyto.git
cd PhasePhyto
pip install -e ".[all]"
```

Verify the install:
```bash
make verify
```

This runs lint + type check + all tests. Current expected result is Ruff passing,
mypy reporting no source issues, and pytest reporting 30 passing tests.

### Step 2: Get Data

**Quick test with synthetic data:**
```bash
make data-synthetic
```
This creates `data/synthetic/plantvillage/` (500 images) and `data/synthetic/plantdoc/` (100 images) with class-specific patterns.

**Real data:**
```bash
# PlantVillage (requires Kaggle API credentials at ~/.kaggle/kaggle.json)
make data-plantvillage

# PlantDoc
make data-plantdoc

# Or both at once
make data-all
```

### Step 3: Train

```bash
# On synthetic data (quick test, ~2 min on GPU)
python -m phasephyto.train --config configs/base.yaml \
    --override data.source_dir=data/synthetic/plantvillage \
    --override data.target_dir=data/synthetic/plantdoc \
    --override training.epochs=10

# On real PlantVillage -> PlantDoc
make train CONFIG=configs/plant_disease.yaml

# Train the semantic-only timm baseline on the same split
make train-baseline CONFIG=configs/plant_disease.yaml
```

### Step 4: Evaluate

```bash
make evaluate CONFIG=configs/plant_disease.yaml CKPT=checkpoints/plant_disease/best_model.pt

# Evaluate the semantic-only baseline for an apples-to-apples comparison
make evaluate-baseline CONFIG=configs/plant_disease.yaml \
    CKPT=checkpoints/plant_disease/baseline/best_model.pt
```

### Step 5: Inference on Your Images

```bash
# Single image
python -m phasephyto.inference \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/plant_disease/best_model.pt \
    --input path/to/leaf_photo.jpg \
    --gradcam

# Batch (entire folder)
python -m phasephyto.inference \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/plant_disease/best_model.pt \
    --input path/to/images/ \
    --gradcam
```

Output goes to `inference_output/` -- one PNG per image showing the original, 3 PC maps, and Grad-CAM overlay.

---

## Getting the Data

### PlantVillage (Source Domain -- Lab)
- **What**: 54,305 images, 38 crop-disease classes, controlled lighting
- **Where**: [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Size**: ~1.5 GB
- **Format**: `plantvillage/<class_name>/*.jpg`

### PlantDoc (Target Domain -- Field)
- **What**: 2,598 images, 27 classes, real-world field conditions
- **Where**: [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)
- **Size**: ~300 MB
- **Format**: `plantdoc/<class_name>/*.jpg`

### Other Datasets (Future Use Cases)

| Dataset | Use Case | Source | Notes |
|---------|----------|--------|-------|
| Potato Tuber Histology | Cross-stain | Contact dataset authors | 3 stains: Safranin-O, Toluidine Blue-O, Lugol's Iodine |
| Pollen Grain | Microscopy | Various palynology databases | 46 categories |
| XyloTron | Wood anatomy | USDA Forest Service | 15 Ghanaian species, lab + field splits |

For now, **focus on PlantVillage + PlantDoc** -- this is the primary benchmark.

Before benchmarking, audit class overlap:

```bash
python scripts/audit_class_overlap.py \
    --source data/plant_disease/plantvillage \
    --target data/plant_disease/plantdoc \
    --fail-on-empty
```

---

## Training PhasePhyto

### What Happens During Training

1. Each image passes through **3 parallel streams**:
   - **PC Stream**: Converts to grayscale, applies 24 Log-Gabor filters via FFT, extracts 3 structural maps (edges, symmetry, texture direction), encodes to 49 tokens
   - **ViT Stream**: Feeds raw RGB to ViT-B/16, extracts 196 patch tokens
   - **Illumination Stream**: Feeds CLAHE-preprocessed image to shallow CNN, extracts 1 auxiliary vector

2. **Cross-attention** fuses streams: structural tokens (Q) query semantic tokens (K,V)
3. Fused features + illumination vector go through classifier head
4. Focal loss handles class imbalance; cosine annealing manages learning rate

### Recommended Configs by Hardware

| GPU | Batch Size | Backbone | Expected Time (PlantVillage, 30 epochs) |
|-----|-----------|----------|------------------------------------------|
| T4 (16GB) | 16 | vit_base_patch16_224 | ~2 hours |
| A100 (40GB) | 64 | vit_base_patch16_224 | ~30 min |
| RTX 3090 (24GB) | 32 | vit_base_patch16_224 | ~1 hour |
| CPU only | 8 | vit_small_patch16_224 | ~12 hours (not recommended) |

If you get OOM errors, reduce batch size:
```bash
python -m phasephyto.train --config configs/plant_disease.yaml --override training.batch_size=8
```

### Config Reference

Edit `configs/plant_disease.yaml` or override from CLI:

```yaml
model:
  backbone_name: "vit_base_patch16_224"  # or vit_small_patch16_224 for less VRAM
  fusion_dim: 256
  freeze_backbone: false  # set true for linear-probe (faster, less VRAM)

training:
  lr: 5.0e-5         # lower for ViT than CNNs
  epochs: 30
  batch_size: 32      # reduce if OOM
  patience: 8         # early stopping
  loss: "focal"       # focal | cross_entropy | label_smoothing
```

---

## Evaluating Domain Shift

The whole point of PhasePhyto is **zero-shot generalisation**: train on lab data, test on field data without fine-tuning.
Training uses source-domain validation only. If the source dataset has no
explicit validation split, the training CLI creates a deterministic source-only
random split using `data.val_split`; PlantDoc is reserved for final evaluation.

### What the Evaluation Measures

| Metric | What It Tells You |
|--------|-------------------|
| **Source Accuracy** | How well the model learned the training distribution |
| **Target Accuracy** | How well it generalises to unseen conditions (the metric that matters) |
| **Accuracy Delta** | Target - Source. Closer to 0 = better generalisation. Compare against the same-split baseline. |
| **F1 Macro** | Class-balanced performance (handles imbalanced PlantDoc) |

### Interpreting Results

```
DOMAIN SHIFT EVALUATION: plant_disease
================================================================
Metric               Source          Target          Delta
-----------------------------------------------------------------
Accuracy             0.9850          0.9210          -0.0640
F1 (macro)           0.9820          0.9050          -0.0770
```

- **Delta of -6.4%** means the model loses 6.4% accuracy moving from lab to field
- Literature baselines can lose substantial accuracy on this benchmark, but
  treat the exact drop as experiment-dependent.
- Do not claim a PhasePhyto improvement until PhasePhyto and the semantic-only
  baseline have both been trained/evaluated on the same source/target split.

### Compare Against Baselines

The Colab notebook trains both PhasePhyto and a standard ViT baseline
side-by-side. The local repo also includes:

```bash
python -m phasephyto.train_baseline --config configs/plant_disease.yaml
python -m phasephyto.evaluate_baseline \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/plant_disease/baseline/best_model.pt
```

Use the resulting source/target deltas to decide whether PhasePhyto is improving
generalization for your specific dataset split.

To run the full local benchmark orchestration and write a Markdown/JSON summary:

```bash
python scripts/benchmark.py \
    --config configs/plant_disease.yaml \
    --output-dir benchmark_results
```

---

## Running Inference on Your Own Images

### What You Need
- A trained checkpoint (`.pt` file from training)
- One or more plant leaf images (JPG/PNG)

### What You Get

For each input image, the inference script generates a **5-panel analysis**:

```
┌──────────┬──────────────┬────────────────┬─────────────────┬───────────┐
│ Original │ PC Magnitude │ Phase Symmetry │ Oriented Energy │ Grad-CAM  │
│          │ (edges)      │ (circles/pores)│ (veins/lines)   │ (overlay) │
└──────────┴──────────────┴────────────────┴─────────────────┴───────────┘
```

### Reading the PC Maps

**PC Magnitude** (hot colormap):
- Bright regions = strong structural edges
- Shows lesion boundaries, leaf margins, vein outlines
- Key property: tested for positive brightness-scaling invariance, so it should
  preserve structural edges across sun/shadow changes better than raw intensity
  gradients.

**Phase Symmetry** (magma colormap):
- Bright regions = symmetric structures
- Detects circular features: spore bodies, stomata, necrotic spots
- Useful for fungal infection identification

**Oriented Energy** (viridis colormap):
- Bright regions = strong directional texture
- Shows vein networks, fungal hyphae, rust streaks
- Anisotropic -- captures directionality that magnitude misses

**Grad-CAM** (jet overlay):
- Red/yellow = where the model is "looking" to make its decision
- Should align with lesion regions, not background
- If Grad-CAM highlights shadows or background, the model may be cheating

---

## Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix**: Reduce batch size.
```bash
python -m phasephyto.train --config configs/plant_disease.yaml --override training.batch_size=8
```
Or use a smaller backbone:
```bash
python -m phasephyto.train --config configs/base.yaml --override model.backbone_name=vit_small_patch16_224
```
Or freeze the backbone (linear probe):
```bash
python -m phasephyto.train --config configs/base.yaml --override model.freeze_backbone=true
```

### No Module Named 'phasephyto'
```
ModuleNotFoundError: No module named 'phasephyto'
```
**Fix**: Run from the project root directory, and make sure you installed with:
```bash
pip install -e .
```

### Kaggle API Not Found
```
OSError: Could not find kaggle.json
```
**Fix**: 
1. Go to kaggle.com > Account > Create New API Token
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

### Tests Fail with Import Errors
```bash
# Make sure you're in the project root
cd PhasePhyto
pip install -e ".[dev]"
python -m pytest tests/ -v
```

### Training Loss Not Decreasing
- Check learning rate: ViT needs lower LR (1e-5 to 5e-5) than CNNs
- Check data: run `make data-synthetic` first to verify pipeline
- Try unfreezing backbone if it was frozen
- Check for NaN: if PC epsilon is too small, gradients explode

### Notebook Cells Fail on Colab
- Make sure you selected **T4 GPU** runtime
- Run cells **in order** from top to bottom (cells depend on earlier definitions)
- If "Session crashed", reduce `batch_size` in the CONFIG cell to 8

---

## Workflow Cheat Sheet

### First Time Setup (5 min)
```bash
git clone <repo>
cd PhasePhyto
pip install -e ".[all]"
make verify              # confirm everything works
make data-synthetic      # generate test data
```

### Quick Pipeline Test (5 min)
```bash
python -m phasephyto.train --config configs/base.yaml \
    --override data.source_dir=data/synthetic/plantvillage \
    --override data.target_dir=data/synthetic/plantdoc \
    --override training.epochs=5 \
    --override training.batch_size=8
```

### Full Training Run (1-2 hours on T4)
```bash
make data-all            # download real data
make train CONFIG=configs/plant_disease.yaml
make train-baseline CONFIG=configs/plant_disease.yaml
```

### Evaluate + Visualise
```bash
make evaluate CONFIG=configs/plant_disease.yaml CKPT=checkpoints/plant_disease/best_model.pt
make evaluate-baseline CONFIG=configs/plant_disease.yaml \
    CKPT=checkpoints/plant_disease/baseline/best_model.pt

python -m phasephyto.inference \
    --config configs/plant_disease.yaml \
    --checkpoint checkpoints/plant_disease/best_model.pt \
    --input data/plant_disease/plantdoc/ \
    --gradcam
```

### Colab Quick Run
1. Upload notebook to Colab
2. Set T4 GPU runtime
3. Run all cells (synthetic mode)
4. Review outputs: training curves, confusion matrices, PC map visualisations
5. Switch to real data when satisfied pipeline works
