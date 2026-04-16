# PhasePhyto Implementation Roadmap

## Current Verification Snapshot

**Last verified local state**: `make verify` passes on Python 3.11.5.

- Ruff: all checks passed.
- mypy: no issues in 28 source files.
- pytest: 30 tests passed, including phase-congruency invariance, orientation
  selectivity, circle-boundary phase symmetry, source-only validation handling,
  model forward/backward, attention normalization, and tiny synthetic training
  smoke tests.
- Colab notebook is present (`notebooks/PhasePhyto_Colab.ipynb`, 48 cells) and
  has been synced to the current `torch.amp` AMP API, corrected Log-Gabor FFT
  grid/sign masks, source-only validation protocol, Colab requirements, and
  PlantDoc `test/` split resolution. It now also supports segregated run
  artifacts under `runs/<run_name>/{checkpoints,plots,results}` with Drive,
  Colab-local SSD, or mounted external/hooked SSD storage backends.
- Downloader notebook can create Drive tar archives (`plantvillage.tar`,
  `plantdoc.tar`), and the training notebook can hydrate `/content/data` from
  those archives to avoid slow Drive file-by-file reads.
- One-time data preparation notebook is present
  (`notebooks/PhasePhyto_Download_Data_To_Drive.ipynb`) to download PlantVillage
  and PlantDoc into Google Drive with a reusable dataset manifest.
- Notebook quality audit passes for JSON validity and code-cell syntax. The
  training notebook now saves a label-aligned target classification report and
  uses explicit labels for PlantDoc confusion matrices.
- Colab troubleshooting docs include corrupt/unreadable image scanning,
  Drive quarantine, and tar archive recreation after dataset cleanup.
- Real-data benchmark numbers are **not yet recorded** in this repo; compare
  PhasePhyto and the semantic-only baseline on the same split before making
  claims about field performance.
- Overall maturity: verified implementation scaffold / pre-real-benchmark
  artifact. Next scientific milestone is to run `scripts/audit_class_overlap.py`
  and `scripts/benchmark.py` on downloaded PlantVillage -> PlantDoc data.

## Phase 0: Project Scaffolding & Infrastructure
**Goal**: Reproducible dev environment, CI-ready project structure.
**Status**: COMPLETE

### 0.1 Repository Setup
- [x] Initialize git repo for this local checkout; `.gitignore` excludes data,
  checkpoints, caches, `.omx/`, benchmark outputs, and local artifacts
- [x] Create project directory structure:
  ```
  PhasePhyto/
  ├── CLAUDE.md
  ├── ROADMAP.md
  ├── pyproject.toml          # single source of truth for deps
  ├── Makefile                # lint, typecheck, test, train, evaluate, data targets
  ├── configs/
  │   ├── base.yaml           # shared hyperparams
  │   ├── plant_disease.yaml
  │   ├── histology.yaml
  │   ├── pollen.yaml
  │   └── wood.yaml
  ├── phasephyto/
  │   ├── __init__.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── phase_congruency.py    # Log-Gabor filter bank + PC computation
  │   │   ├── pc_encoder.py          # 2-layer CNN: PC maps -> Structural Tokens
  │   │   ├── semantic_backbone.py   # ViT-B/16 backbone wrapper (timm)
  │   │   ├── illumination_norm.py   # CIELAB + CLAHE + shallow CNN
  │   │   ├── cross_attention.py     # Fusion: Q=structural, K/V=semantic
  │   │   └── phasephyto.py          # Full model assembly
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── datasets.py            # Dataset classes per use case + TransformSubset
  │   │   ├── splits.py              # Split resolution + class-count helpers
  │   │   └── transforms.py          # DualTransform (RGB+CLAHE), augmentations
  │   ├── training/
  │   │   ├── __init__.py
  │   │   ├── trainer.py             # Training loop with AMP, grad clip, early stopping
  │   │   └── losses.py              # CE, focal loss, label smoothing
  │   ├── evaluation/
  │   │   ├── __init__.py
  │   │   ├── metrics.py             # Accuracy, F1, confusion matrix, per-class
  │   │   ├── domain_shift.py        # Cross-domain evaluation protocol
  │   │   └── xai.py                 # Grad-CAM, attention visualization
  │   ├── utils/
  │   │   ├── __init__.py
  │   │   ├── config.py              # YAML config loader with dataclass validation
  │   │   └── seed.py                # Reproducibility (torch, numpy, cudnn)
  │   ├── train.py                   # Entry point: PhasePhyto training
  │   ├── train_baseline.py          # Entry point: semantic-only baseline training
  │   ├── evaluate.py                # Entry point: evaluation (all 4 use cases)
  │   ├── evaluate_baseline.py       # Entry point: baseline domain-shift evaluation
  │   └── inference.py               # Entry point: single image / batch + PC visualisation
  ├── tests/
  │   ├── test_data_protocol.py      # Source-only validation + split resolution
  │   ├── test_phase_congruency.py   # Filter bank, PC maps, amplitude invariance
  │   ├── test_model_forward.py      # Forward/backward pass, shapes, param counts
  │   └── test_training_smoke.py     # Tiny synthetic trainer + baseline smoke tests
  ├── notebooks/
  │   ├── README.md                  # Notebook workflow index
  │   ├── PhasePhyto_Download_Data_To_Drive.ipynb # One-time Drive dataset prep
  │   ├── PhasePhyto_Colab.ipynb     # Training/evaluation notebook (48 cells)
  │   ├── PhasePhyto_Inspect_00_Index.ipynb # Inspection index/run chooser
  │   ├── PhasePhyto_Inspect_01_Run_Overview.ipynb # Manifest/artifact viewer
  │   ├── PhasePhyto_Inspect_02_Metrics.ipynb # Metrics comparison
  │   ├── PhasePhyto_Inspect_03_Plots.ipynb # Plot/image viewer
  │   ├── PhasePhyto_Inspect_04_Reports.ipynb # Report viewer
  │   └── PhasePhyto_Inspect_Run_Results.ipynb # Compatibility pointer
  └── scripts/
      ├── audit_class_overlap.py     # PlantVillage/PlantDoc overlap audit
      ├── benchmark.py               # PhasePhyto-vs-baseline benchmark orchestration
      └── download_data.py           # PlantVillage (Kaggle), PlantDoc (GitHub), synthetic
  ```
- [x] `pyproject.toml` with pinned dependencies:
  - torch>=2.1, torchvision, timm, numpy, scipy, opencv-python-headless, scikit-learn, pyyaml, wandb, matplotlib, ruff, mypy, pytest

### 0.2 Config System
- [x] YAML-based config with dataclass validation (`configs/base.yaml`)
- [x] Hierarchical override: `base.yaml` -> `<use_case>.yaml` -> CLI args
- [x] Key config fields: `model.backbone_name`, `model.pc_scales`, `model.pc_orientations`, `model.fusion_dim`, `training.lr`, `training.epochs`, `training.batch_size`, `data.root`, `data.use_case`

### 0.3 CI Basics
- [ ] Pre-commit hooks: ruff format, ruff check, mypy
- [x] GitHub Actions: lint + type-check + unit tests on push
- [x] Makefile with targets: `lint`, `lint-fix`, `typecheck`, `test`,
  `test-pc`, `test-model`, `verify`, `train`, `train-baseline`, `evaluate`,
  `evaluate-baseline`, `audit-classes`, `benchmark`, `inference`,
  `data-synthetic`, `data-plantvillage`, `data-plantdoc`, `data-all`, `clean`

---

## Phase 1: Core Phase Congruency Module
**Goal**: Standalone, differentiable, GPU-accelerated PC computation with correctness tests.
**Critical path**: Everything downstream depends on this.
**Status**: COMPLETE

### 1.1 Log-Gabor Filter Bank (`phasephyto/models/phase_congruency.py`)
- [x] Implement `LogGaborFilterBank(nn.Module)`:
  - Constructor params: `num_scales=4`, `num_orientations=6`, `image_size=(224, 224)`, `min_wavelength=3`, `mult=2.1`, `sigma_on_f=0.55`, `d_theta_on_sigma=1.2`
  - Pre-compute 24 filters in frequency domain during `__init__`
  - Register all filters as `register_buffer` (not Parameters)
  - Forward: `(B, 1, H, W) -> even (B, 24, H, W), odd (B, 24, H, W)` via `torch.fft.rfft2` + pointwise multiply + `torch.fft.irfft2`
  - Oriented Hilbert transform for quadrature (odd) responses via per-filter sign masks
  - Frequency grid matches the unshifted `torch.fft.rfft2` layout (DC at `[0, 0]`)
- [ ] **Test**: Verify filter bank energy conservation (Parseval's theorem)
- [x] **Test**: Verify `PC(image) == PC(image * k)` for k in [0.1, 0.5, 2.0, 5.0, 10.0] within tolerance 5e-3

### 1.2 Phase Congruency Maps
- [x] Implement `PhaseCongruencyExtractor(nn.Module)`:
  - Input: even + odd filter responses `(B, 24, H, W)` each
  - Output dict: `{"pc_magnitude": (B, 1, H, W), "phase_symmetry": (B, 1, H, W), "oriented_energy": (B, 1, H, W)}`
  - Noise threshold estimation via median of finest-scale responses
  - Frequency spread weighting function with sigmoid cutoff
  - Epsilon = 1e-6 minimum in denominator
  - Border taper suppresses FFT circular wrap artifacts in structural maps
- [x] **Test**: PC magnitude of a synthetic step edge should have peak at edge location
- [x] **Test**: Phase symmetry of a synthetic circle should have response at boundary
- [x] **Test**: Oriented energy of horizontal lines should peak at 0-degree orientation bin
- [x] **Test**: All outputs in range [0, 1]
- [x] **Test**: No NaN in outputs

### 1.3 PC Encoder
- [x] Implement `PCEncoder(nn.Module)`:
  - Input: 3-channel concatenated PC maps `(B, 3, H, W)`
  - Two Conv2d layers with BatchNorm + GELU
  - Adaptive average pool to `(7, 7)` spatial
  - Output: Structural Tokens `(B, 49, 256)` for attention
- [x] **Test**: Forward pass shape correctness

---

## Phase 2: Backbone Streams
**Goal**: Semantic backbone and illumination-normalized stream producing compatible token formats.
**Status**: COMPLETE

### 2.1 Semantic Backbone (`phasephyto/models/semantic_backbone.py`)
- [x] Implement `SemanticBackbone(nn.Module)`:
  - Wraps any `timm` model (default: `vit_base_patch16_224`)
  - For ViT: extracts 196 patch tokens (14x14), drops CLS/prefix tokens
  - For CNN: extracts spatial features, flattens to token sequence
  - Linear projection + LayerNorm to fusion_dim (256)
  - Output: Semantic Tokens `(B, 196, 256)` for ViT
- [x] Support swappable backbones via config: `vit_base_patch16_224`, `vit_small_patch16_224`, `efficientnet_b0`, `convnext_tiny`, `resnet50`
- [ ] **Test**: Output shape matches for all supported backbones

### 2.2 Illumination Normalization (`phasephyto/models/illumination_norm.py`)
- [x] Implement `IlluminationNormStream(nn.Module)`:
  - CLAHE applied as preprocessing transform via `CLAHEPreprocessor` (non-differentiable, not in forward)
  - 2-layer shallow CNN + AdaptiveAvgPool -> auxiliary feature vector `(B, 256)`
- [ ] **Test**: CIELAB conversion matches OpenCV `cv2.cvtColor` within tolerance
- [ ] **Test**: Output vector shape correctness

### 2.3 Integration Test
- [x] All three streams produce compatible shapes
- [x] Full forward pass with random input `(B, 3, 224, 224)` completes without error
- [x] Gradient flows through all learnable parameters (no detached paths)

---

## Phase 3: Cross-Attention Fusion & Full Model
**Goal**: Assembled PhasePhyto model, end-to-end differentiable, ~331K fusion params.
**Status**: COMPLETE

### 3.1 Cross-Attention Fusion (`phasephyto/models/cross_attention.py`)
- [x] Implement `StructuralSemanticFusion(nn.Module)`:
  - Multi-head cross-attention (4 heads, dim=256) via `nn.MultiheadAttention`
  - Q = Structural Tokens (49) from PC Encoder
  - K, V = Semantic Tokens (196) from ViT backbone
  - Pre-LayerNorm on Q and K/V, post-LayerNorm on output
  - Residual connection + lightweight FFN (default `dim -> dim/2 -> dim`) with GELU
  - Output: mean-pooled fused features `(B, 256)`
- [x] **Test**: Attention weights sum to 1 along key dimension
- [x] **Test**: Parameter count of fusion module is ~331K (150K-350K range)
- [ ] **Test**: Grad-CAM on Q-K attention map highlights structural edges, not flat regions

### 3.2 Full Model Assembly (`phasephyto/models/phasephyto.py`)
- [x] Implement `PhasePhyto(nn.Module)`:
  - Composes all sub-modules
  - Classification head: `Linear(512, 256) -> GELU -> Dropout -> Linear(256, num_classes)` (fused + illumination auxiliary)
  - Config-driven: backbone choice, num_classes, PC params
  - `forward(x_rgb, x_clahe) -> dict with 'logits' (B, num_classes)`
  - Optional `return_maps=True` returns PC maps, `return_attention=True` returns attention weights
  - `count_parameters()` method for per-module breakdown
- [x] **Test**: End-to-end forward + backward on random data
- [x] **Test**: `model.eval()` + `torch.no_grad()` inference path
- [x] **Test**: Model parameter count breakdown (filter_bank=0, pc_extractor=0, etc.)
- [ ] **Test**: ONNX export feasibility check

---

## Phase 4: Data Pipeline
**Goal**: Dataset classes, download scripts, and augmentation pipelines for all 4 use cases.
**Status**: COMPLETE

### 4.1 PlantVillage / PlantDoc (Use Case 1)
- [x] Download script for PlantVillage (Kaggle API) in `scripts/download_data.py`
- [x] Download script for PlantDoc (GitHub clone) in `scripts/download_data.py`
- [x] Synthetic data generator with class-specific frequency patterns + brightness domain shift
- [x] `PlantDiseaseDataset(Dataset)`: handles both sources, returns `(rgb, clahe, label)` with DualTransform
- [x] Class mapping: `class_to_idx` parameter for aligning PlantDoc to PlantVillage classes
- [x] Train/val split on PlantVillage with `TransformSubset` fix; PlantDoc is test-only (OOD)
- [x] Standard augmentations: RandomResizedCrop(224), HorizontalFlip, VerticalFlip, ColorJitter, Normalize

### 4.2 Potato Tuber Histology (Use Case 2)
- [x] `HistologyDataset` with stain-based filtering (`safranin`, `toluidine`, `lugol`, `all`)
- [x] Labels from directory structure (physical/morphological/tissue grading)
- [x] Cross-stain evaluation protocol via config (`data.stain` parameter)
- [ ] Microscopy-specific augmentations: rotation (0/90/180/270), elastic deformation, stain jitter

### 4.3 Pollen Grain (Use Case 3)
- [x] `PollenDataset` with standard image-folder layout (46 categories)
- [ ] Non-local means denoising as optional preprocessing
- [ ] Gamma correction normalization
- [ ] Cross-microscope evaluation split (if metadata available)

### 4.4 Wood Anatomy / XyloTron (Use Case 4)
- [x] `WoodDataset` with domain-based filtering (`lab`, `field`, `all`)
- [x] Lab vs. field split for domain shift evaluation
- [ ] Augmentations: rotation, brightness jitter, Gaussian blur for sanding quality

### 4.5 Unified DataModule
- [x] `DATASET_MAP` dispatch in `train.py` and `evaluate.py` selects correct dataset from config
- [x] `DualTransform` provides consistent `(rgb_tensor, clahe_tensor)` interface across all datasets
- [x] `TransformSubset` utility for safe `random_split` with transform override
- [ ] **Test**: Each dataset loads, returns correct shapes and label ranges

### 4.6 Bug Fixes (discovered during audit)
- [x] **CRITICAL**: Fixed `random_split` val subset inheriting train augmentations via `TransformSubset`
- [x] **CRITICAL**: Fixed `evaluate.py` hardcoded to `PlantDiseaseDataset` -- now uses `DATASET_MAP` for all 4 use cases

---

## Phase 5: Training Infrastructure
**Goal**: Complete training loop with logging, checkpointing, and reproducibility.
**Status**: COMPLETE

### 5.1 Training Loop (`phasephyto/training/trainer.py`)
- [x] `Trainer` class:
  - Mixed precision (`torch.amp`) with GradScaler
  - Gradient clipping (max_norm=1.0, configurable)
  - Cosine annealing LR scheduler with warm restarts
  - Linear warmup for first N epochs
  - Best model checkpointing (by validation F1)
  - Early stopping (patience configurable, default=10)
  - Optional wandb logging: loss, accuracy, F1, LR
- [x] Reproducibility: `seed_everything()` at script entry (torch, numpy, cudnn.deterministic)
- [x] Tiny synthetic CPU training smoke test (`tests/test_training_smoke.py`)

### 5.2 Loss Functions (`phasephyto/training/losses.py`)
- [x] CrossEntropyLoss (baseline, via torch.nn)
- [x] Focal Loss (for class imbalance, gamma=2.0 default)
- [x] Label Smoothing CE (smoothing=0.1 default)
- [x] Config-selectable via `training.loss` field

### 5.3 Entry Points
- [x] `phasephyto/train.py`: `python -m phasephyto.train --config configs/plant_disease.yaml --override training.lr=1e-4`
- [x] `phasephyto/evaluate.py`: load checkpoint, run source + target domain metrics, all 4 use cases
- [x] `phasephyto/inference.py`: single image or directory, outputs predictions + PC map visualization + optional Grad-CAM
- [x] `phasephyto/train_baseline.py`: train semantic-only timm baseline with the same loaders/losses
- [x] `phasephyto/evaluate_baseline.py`: evaluate baseline on the same source + target domain protocol

---

## Phase 6: Baseline Benchmarking
**Goal**: Reproduce the baseline numbers from the instructions to establish ground truth before PhasePhyto evaluation.
**Status**: PARTIAL (Colab notebook plus CLI baseline path implemented; real-data numbers pending)

### 6.1 Baseline Models on PlantVillage -> PlantDoc
- [x] Baseline ViT-B/16 comparison implemented in Colab notebook (Section 9)
- [x] Colab notebook installs required notebook dependencies and runs synthetic
  PhasePhyto-vs-baseline comparison by default before real-data runs
- [x] Semantic-only timm baseline model (`phasephyto/models/baseline.py`)
- [x] CLI baseline training/evaluation (`phasephyto/train_baseline.py`, `phasephyto/evaluate_baseline.py`)
- [x] Baseline output-contract smoke test
- [ ] Train ResNet-50 on PlantVillage, evaluate OOD on PlantDoc
- [ ] Train EfficientNet-B0 on PlantVillage, evaluate OOD on PlantDoc
- [ ] Train ConvNeXt-Tiny on PlantVillage, evaluate OOD on PlantDoc
- [ ] Record: in-distribution accuracy, OOD F1, accuracy delta
- [ ] **Target**: Reproduce ~70-75% PlantDoc F1 range from literature

### 6.2 Baseline on Cross-Stain Histology
- [ ] Train VGG16 on single-stain potato tuber data
- [ ] Evaluate cross-stain transfer (6 permutations)
- [ ] Record per-stain accuracy and cross-stain degradation

### 6.3 Results Table
- [x] Auto-generate comparison table (baselines vs. PhasePhyto) via `scripts/benchmark.py`
- [ ] Record PhasePhyto vs. baseline metrics in README/GUIDE after real-data run
- [ ] wandb dashboard for all experiments

---

## Phase 7: PhasePhyto Training & Evaluation
**Goal**: Train the full PhasePhyto model on all 4 use cases, demonstrate domain-shift resilience.
**Status**: PENDING (infrastructure ready, awaiting real data)

### 7.1 Use Case 1: Plant Disease (Lab-to-Field)
- [ ] Train PhasePhyto on PlantVillage
- [ ] Evaluate on PlantDoc (zero-shot OOD)
- [ ] **Target**: PlantDoc F1 > 90% (delta < -5% from in-distribution)
- [ ] Ablation: PC stream only, backbone only, full fusion
- [ ] Ablation: contribution of each PC map (magnitude, symmetry, oriented energy)

### 7.2 Use Case 2: Cross-Stain Histology
- [ ] Train on Safranin-O, test on Toluidine Blue-O and Lugol's Iodine (and all permutations)
- [ ] **Target**: Cross-stain accuracy drop < 5% compared to within-stain
- [ ] Visualize PC maps showing stain-invariant cell wall detection

### 7.3 Use Case 3: Pollen Classification
- [ ] Train on curated microscopy data
- [ ] Evaluate cross-microscope generalization
- [ ] **Target**: Maintain 98%+ accuracy across varied microscope configurations
- [ ] Grad-CAM analysis: verify attention on apertures, not staining artifacts

### 7.4 Use Case 4: Wood Anatomy
- [ ] Train on XyloTron lab specimens
- [ ] Evaluate on field specimens
- [ ] **Target**: Recover majority of the 25% lab-to-field accuracy gap
- [ ] Visualize vessel boundary detection via PC magnitude

### 7.5 Cross-Use-Case Analysis
- [ ] Compare PhasePhyto's generalization gap across all 4 domains
- [ ] Statistical significance tests (paired t-test / Wilcoxon across random seeds)
- [ ] Generate publication-ready figures and tables

---

## Phase 8: Explainability & Visualization
**Goal**: Prove the model is using physics-based features, not shortcuts.
**Status**: PARTIAL (Grad-CAM + PC viz implemented, pending real-data analysis)

### 8.1 Grad-CAM Integration
- [x] Implement `GradCAMPhasePhyto` for the cross-attention layer (`phasephyto/evaluation/xai.py`)
- [x] Visualize which spatial regions the structural queries attend to (Colab notebook Section 8)
- [x] Side-by-side: raw image, PC magnitude, phase symmetry, oriented energy, attention heatmap

### 8.2 PC Map Visualization
- [x] Utility to render all 3 PC maps for any input image (`visualize_attention()` in `xai.py`)
- [x] Overlay mode: attention heatmap on original image
- [x] Compare PC maps under different lighting conditions (Colab notebook "Illumination Invariance Demo")

### 8.3 Failure Case Analysis
- [ ] Identify and catalog failure modes:
  - Low SNR (noisy sensors, motion blur)
  - Overlapping/occluded leaves
  - Periodic background textures (greenhouse mesh)
  - JPEG compression artifacts
- [ ] Per-failure-mode accuracy breakdown

---

## Phase 9: Optimization & Edge Deployment
**Goal**: Production-ready inference pipeline for mobile/edge hardware.
**Status**: PENDING

### 9.1 Model Optimization
- [ ] TorchScript / `torch.compile()` for the full pipeline
- [ ] ONNX export with custom FFT op handling
- [ ] Quantization analysis: INT8 for backbone, FP16 for PC stream (phase precision matters)
- [ ] Benchmark: latency, throughput, memory on NVIDIA GPU

### 9.2 Jetson Nano Deployment
- [ ] TensorRT conversion with cuFFT integration for Log-Gabor filtering
- [ ] Benchmark: target < 50ms per image
- [ ] Memory profiling: fit within 4GB unified memory
- [ ] Power consumption measurement

### 9.3 Mobile Deployment (Stretch)
- [ ] CoreML export (iOS) / ONNX Runtime Mobile (Android)
- [ ] CPU-GPU co-execution for FFT operations on mobile SoC
- [ ] Benchmark on representative smartphone hardware
- [ ] Target: < 2 second inference on mid-tier device

---

## Phase 10: Paper & Release
**Goal**: Reproducible research artifact ready for submission.
**Status**: PARTIAL (README, GUIDE, configs, and Colab done; Docker/checkpoint hosting pending)

### 10.1 Documentation
- [x] README with installation, quickstart, architecture diagram, and reproduction instructions
- [x] Colab notebook as primary evaluation notebook
- [x] Guide with setup, data download, training, evaluation, inference, and troubleshooting
- [x] Roadmap updated with verified status and remaining real-data validation tasks
- [ ] Per-use-case standalone evaluation notebooks
- [ ] Architecture diagrams (draw.io or TikZ)

### 10.2 Reproducibility Package
- [x] All configs checked in
- [ ] Trained checkpoint hosting (HuggingFace Hub or Zenodo)
- [ ] `scripts/reproduce.sh` that runs all experiments end-to-end
- [ ] Docker container with pinned environment

### 10.3 Release
- [x] Installable via `pip install -e .` (pyproject.toml)
- [ ] PyPI package
- [ ] HuggingFace model card
- [x] License: Apache-2.0

---

## Dependency Graph (Critical Path)

```
Phase 0 (Scaffolding)         ✅ COMPLETE
  └─> Phase 1 (PC Module)     ✅ COMPLETE
        ├─> Phase 2 (Streams)  ✅ COMPLETE
        │     └─> Phase 3 (Fusion) ✅ COMPLETE
        │           └─> Phase 5 (Training) ✅ COMPLETE
        │                 ├─> Phase 6 (Baselines)  ⚠️ PARTIAL (CLI + Colab ready)
        │                 └─> Phase 7 (Training)   ⏳ PENDING (needs real data)
        │                       └─> Phase 8 (XAI)  ⚠️ PARTIAL
        │                             └─> Phase 10 (Paper) ⚠️ PARTIAL
        └─> Phase 4 (Data)    ✅ COMPLETE
                                    Phase 9 (Edge)  ⏳ PENDING
```

## Risk Register

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Log-Gabor filter numerical instability during backprop | Blocks all training | Match FFT grid layout, clamp denominators, test gradient flow early (Phase 1) | **Mitigated** -- invariance/no-NaN/step-edge tests pass |
| PlantVillage/PlantDoc class mismatch | Invalid benchmark | Map classes carefully in Phase 4, document excluded classes | **Mitigated in tooling** -- `class_to_idx` alignment plus `scripts/audit_class_overlap.py`; real data still must be audited before claims |
| CLAHE non-differentiability breaks end-to-end training | Reduced gradient signal to illumination stream | Apply CLAHE as preprocessing transform, not in computational graph | **Mitigated** -- CLAHE in `DualTransform`, not in forward() |
| FFT on variable-size inputs | Runtime crashes | Enforce fixed input size (224x224) via transforms; pad if needed | **Mitigated** -- all transforms enforce 224x224 |
| Memory OOM with 24 filter responses at full resolution | Cannot train at target resolution | Profile early; consider sequential filter application or half-precision | **Open** -- needs profiling on T4 |
| Dataset availability (some may require institutional access) | Incomplete benchmarking | Identify data sources in Phase 0; have fallback synthetic benchmarks | **Mitigated** -- synthetic data generator in `scripts/download_data.py` |
| `random_split` val subset using train augmentations | Inflated validation metrics | Wrap val subset with `TransformSubset` to override transforms | **Fixed** -- `TransformSubset` in datasets.py, source-only split test, and notebook |
| Overclaiming domain-shift gains before real benchmarks | Misleading release docs | Ship baseline CLIs and keep claims conditional until same-split results exist | **Mitigated in docs** -- real metrics still pending |
| FFT phase quantization sensitivity | Accuracy loss when deploying INT8 on edge | Test FP16/FP32 for PC stream separately from backbone | **Open** -- needs Phase 9 investigation |
