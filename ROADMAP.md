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
- PlantDoc -> PlantVillage alias mapping is implemented so raw PlantDoc class
  names can evaluate against PlantVillage-trained source classes even when exact
  normalized overlap is zero.
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
- [x] Ablation: PC stream only, backbone only, full fusion -- wired into
  `notebooks/PhasePhyto_Colab.ipynb` via `CONFIG["ablation"]` (v0.1.14);
  run-directory auto-suffixed so four sequential runs produce independent
  artifacts. **Ablation table complete as of 2026-04-23**; see `RESULTS.md`:
  `full+leafmask` (target F1 0.3944), `backbone_only` (target F1 0.3859),
  `no_fusion` (target F1 0.3464), `pc_only` (target F1 0.0791). Headline:
  `full` beats `backbone_only` by +0.009 target F1 -- inside noise.
- [ ] Ablation: contribution of each PC map (magnitude, symmetry, oriented energy)

#### 7.1.a OOD Foreground Segmentation & Diagnostic Hooks (v0.1.15)
**Trigger**: 2026-04-21 `pc_only` ablation showed PC stream target F1=0.08
(see `RESULTS.md`). PC features trained on PlantVillage do not transfer to
PlantDoc, most likely because they fit studio-background phase structure
absent in field images. Added a cheap HSV-saturation leaf foreground mask
so the PC stream can be told to ignore background pixels on both train
and target.

- [x] Add CONFIG knobs `leaf_mask_mode`, `leaf_mask_sat_thresh`,
  `leaf_mask_blur_sigma`, `checkpoint_every`, `target_snapshot_every`
  (cells 8 + 10 of training notebook).
- [x] Add `_hsv_leaf_mask()` + `_apply_leaf_mask()` helpers and thread them
  through `DualTransform` so train and val pipelines both gate non-leaf
  pixels when enabled (cell 19).
- [x] Add periodic-checkpoint save and target-domain snapshot inside the
  training loop so OOD trajectory is visible during training instead of
  only at the end (cell 42).
- [x] Run `full` + `leaf_mask_mode="hsv"` + `target_snapshot_every=3`
  (completed 2026-04-21 as `20260421-202641_full`; target F1 = 0.3944).
- [ ] Run `full` + `leaf_mask_mode="hsv_blur"` (pending, after first
  leaf-mask run lands).
- [ ] Evaluate: if leaf masking alone does not lift target, rely on
  Phase 7.1.b pseudo-labeling (below) to add target-side gradient.

#### 7.1.b Pseudo-Label Self-Training (v0.1.16)
**Trigger**: both `full` and `backbone_only` saturate source val >= 0.99,
so additional source-only regularisation cannot help. Only target-side
gradient can close the residual gap. Pseudo-labeling is the smallest
lever that introduces that signal without a full adversarial
architecture (which is held as last resort, see below).

- [x] Add CONFIG knobs `use_pseudo_label`, `pseudo_label_threshold`
  (default 0.9), `pseudo_label_epochs` (default 5), `pseudo_label_lr_mult`
  (default 0.1), `pseudo_label_min_samples` (default 50). (Cells 8, 10.)
- [x] Insert pseudo-label phase as a new cell after the main training
  loop (cell 43 in the patched notebook). Uses EMA model to generate
  labels, filters by confidence threshold, builds a `ConcatDataset` of
  source subset + pseudo-labeled target (with train-transform
  augmentation), fine-tunes for N epochs at `lr * lr_mult`.
- [x] Save pseudo checkpoint as `pseudo_phasephyto.pt`; cell 47 (shifted
  from 46 after insertion) prefers it over best-val / final-EMA when
  present.
- [ ] Run `full` + leaf-mask + pseudo-label (v0.1.16 default) and record
  numbers in `RESULTS.md`.
- [ ] Isolation runs: `full` + leaf-mask only (no pseudo) and `full` +
  pseudo only (no leaf-mask), to attribute gap-closure per lever.

#### 7.1.a/b polish (v0.1.17)
- [x] Confidence-histogram print in the pseudo-label cell (deciles +
  counts above common thresholds) so threshold tuning is visible before
  filtering. Avoids silent "skipped - too few confident samples" runs.
- [x] `aux_pc_weight` hard-zeroed during pseudo-label fine-tune. The PC
  aux head pulls gradients through a non-discriminative pathway on OOD
  and wastes fine-tune budget.
- [x] `history` dict dumped to `RESULTS_DIR/history.json` at end of
  training. Lets post-run inspection notebooks rebuild target-snapshot
  curves even if the kernel dies before the eval cells run.
- [x] `config` snapshot embedded in every `torch.save` call (best,
  periodic, final-EMA, pseudo). Any checkpoint can now be traced back
  to the full CONFIG it was trained under.
- [x] Leaf-mask sanity-viz cell (new, index 52). Renders N source and N
  target images with HSV foreground masks + prints per-image foreground
  fraction. Pre-flight check for HSV threshold drift.
- [x] **Amplitude-invariance regression fix** (cell 26):
  `self.sqrt_eps` was silently changed to `math.sqrt(self.eps)` = 1e-3,
  floor'd the amplitude at a scale-independent constant and broke
  invariance at every k. Restored to `1e-12`; anti-regression rule
  added to `CLAUDE.md`. Numpy repro confirms the fix.

#### 7.1.c DANN / Gradient Reversal (last resort, deferred)
- [ ] **Hold until 7.1.b pseudo-label is fixed and measured.** If
  pseudo-label rerun with a calibrated threshold closes the gap to < 10%,
  DANN is not needed. If it stalls > 20%, add a domain discriminator on
  the fused features with GRL, train adversarially against the
  classifier. Under the pivoted framing (see 7.2 below) a DANN negative
  result is also publishable and stops the lever chase.
- [ ] Plain-ViT-only ablation row (no PhasePhyto PC / fusion code path
  at all) for writeup. Cheap to produce once architecture is frozen.

### 7.2 Close-Out for Pivot to Negative-Study Framing (v0.1.18, 2026-04-22)

**Context.** On 2026-04-22 the project pivoted from "physics-informed
fusion beats a ViT baseline under OOD" to "rigorous negative study of PC
features under domain shift + practical OOD recipe." See `RESULTS.md`
"Project Thesis Pivot" section for the full rationale. This phase
enumerates the minimal remaining experimental work to publish the
negative study honestly.

- [x] **Finish `backbone_only` ablation.** Completed 2026-04-23
  (`20260423-XXXXXX_backbone_only`). Target acc 0.5229, F1 0.3859.
  Outcome (a) landed: `full` (0.5229 / 0.3944) beats `backbone_only`
  (0.5229 / 0.3859) by 0.000 target acc / +0.009 target F1 -- inside
  run-to-run noise. The PC stream + cross-attention does not beat a
  label-smoothed, strongly-augmented ViT under the identical recipe.
  The recipe is the contribution. See RESULTS.md for the detailed
  ablation-table analysis and per-class breakdown.
- [ ] **Rerun Phase 7.1.b pseudo-label with threshold=0.7.** The
  2026-04-23 `no_fusion` run's cell-43 confidence histogram reported
  p50=0.812, p95=0.904 on the 153-sample target loader, with only 19
  samples >=0.9 -- below `pseudo_label_min_samples=50`, so the phase
  auto-skipped. This matches the `full+leafmask` skip behavior. Threshold
  0.7 would admit ~87 samples (empirically justified retry, not a guess).
  Preferably run on top of `backbone_only` given its flatter target
  trajectory across training (0.3785 at epoch 3 -> 0.4027 at epoch 15,
  trending up, vs `full+leafmask` which drifts down).
- [x] **Run `no_fusion` ablation.** Completed 2026-04-23
  (`20260423-132514_no_fusion`). Target F1 = 0.3464 vs `full+leafmask`
  0.3944, so cross-attention buys ~0.048 target F1 under an otherwise
  identical recipe. Modest but non-zero contribution; keep in the
  published recipe. See RESULTS.md for the full ablation-table cell and
  per-class breakdown.
- [ ] **One DANN run (Phase 7.1.c).** Only after pseudo-label is
  confirmed firing. If DANN also fails to move target past ~0.55, the
  negative result is complete.
- [ ] **Optional: best-target-snapshot checkpoint.** Tiny cell-42 patch
  to save `best_target_phasephyto.pt` when the periodic target snapshot
  beats the running best. Does not change the headline metric (still
  selected on source val) but exposes the "best-val != best-target"
  pathology in every future run without dedicated ablations.
- [x] **Write-up pass.** Draft manuscript landed 2026-04-23 as `PAPER.md`.
  Nine sections covering: (1) the OOD-hardening recipe + per-lever deltas,
  (2) the complete four-cell ablation table with interpretation,
  (3) the PC transfer failure as a named failure mode
  (*invariance-classifier-head gap*),
  (4) the selection-on-source-val pathology demonstrated by target
  snapshots across all four ablations,
  (5) limitations (target n=153, missing plain-timm-ViT row, pseudo-label
  not fired, DANN not run),
  (6) references.
  Remaining polish: defensive experiments (plain-timm-ViT row;
  pseudo-label at threshold 0.7) are optional and marked as write-up-stage
  future work in the manuscript rather than blocking submission.

### 7.3 Originally-Planned Use Cases 2-4 (DEPRIORITIZED under pivot)

Code and configs remain in the repo. No experimental work currently
planned. If future work resumes these, the PC claim might recover in
texture-dominated domains (wood anatomy, pollen), but that is a
separate study from the pivoted one.

- [ ] Use Case 2 (cross-stain histology) -- out of scope, see pivot.
- [ ] Use Case 3 (pollen grain classification) -- out of scope, see pivot.
- [ ] Use Case 4 (wood anatomy) -- out of scope, see pivot.

### 7.4 Cross-Use-Case Analysis (DEPRIORITIZED under pivot)

- [ ] Cross-use-case generalization comparison -- out of scope.
- [ ] Statistical significance tests across seeds -- still applies to
  Use Case 1 within the close-out (7.2).
- [ ] Publication figures -- tracked as part of 7.2 write-up pass.

### 7.5 Dataset Expansion (STAGING as of v0.1.19, not active)

Cassava Leaf Disease (Kaggle) is integrated end-to-end in the data
pipeline but no training runs have been executed. The integration is
**staged for future use**, not part of the current 7.2 close-out, and
sits behind a `CONFIG["dataset"]` flag that defaults to `"plantvillage"`
(prior behavior preserved byte-for-byte in control flow).

**Integrated (v0.1.19):**

- [x] `scripts/download_data.py --dataset cassava` reorganizes Kaggle's
  flat `train_images/` + `train.csv` into `<Crop>___<Disease>/` ImageFolder
  layout (~21k images, 5 classes, ~5.5 GB).
- [x] Downloader notebook cell added to produce `cassava.tar` in Drive
  alongside `plantvillage.tar` / `plantdoc.tar`.
- [x] `CassavaDataset(PlantDiseaseDataset)` thin subclass with
  `EXPECTED_CLASSES` tuple; registered in `DATASET_MAP` for both
  `train.py` and `evaluate.py`.
- [x] Training notebook `CONFIG["dataset"]` selector with stratified
  80/10/10 train/val/test split for Cassava (single-domain, so the 10%
  held-out is in-distribution test, not OOD).
- [x] Eval heading + JSON (`phasephyto_domain_shift.json`) context-aware:
  labels Cassava evaluation as "IN-DISTRIBUTION HELD-OUT EVALUATION"
  with `eval_protocol = "in-distribution held-out (Cassava)"`.
- [x] `tests/test_cassava_dataset.py` (3 tests: ImageFolder load,
  DATASET_MAP dispatch, 5-class sanity).

**Not yet done (resume items):**

- [ ] Accept Kaggle competition rules at
  https://www.kaggle.com/competitions/cassava-leaf-disease-classification/rules
  and run `PhasePhyto_Download_Data_To_Drive.ipynb` with
  `download_cassava=True` to materialize `cassava.tar`.
- [ ] First training run on Cassava: set `CONFIG["dataset"] = "cassava"`
  in Colab cell 10 and run top-to-bottom. Establishes baseline
  in-distribution test F1 as a reference number; compare to the full
  OOD recipe without the PC stream.
- [ ] Protocol note in `RESULTS.md` (new run entry) documenting that
  Cassava numbers are in-distribution, not OOD transfer.
- [ ] Optional: identify a true OOD split for Cassava (e.g., train on
  4 classes, evaluate on all 5; or train on one mobile device's images
  and test on another if provenance metadata is available). Requires
  deeper protocol design; not covered by v0.1.19.
- [ ] Optional: follow-on datasets for further expansion (e.g., Plant
  Pathology 2021-FGVC8). Infrastructure already supports adding them via
  the same thin-subclass + `DATASET_MAP` pattern as Cassava.

**Explicit scope constraint:** This dataset expansion is NOT a
replacement for the 7.2 close-out. Finishing `backbone_only` /
`no_fusion` / pseudo-label debug / DANN on PV->PD remains the primary
outstanding work for the pivoted thesis. Cassava is orthogonal runway
for future sessions.

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
Phase 0 (Scaffolding)         COMPLETE
  └─> Phase 1 (PC Module)     COMPLETE
        ├─> Phase 2 (Streams)  COMPLETE
        │     └─> Phase 3 (Fusion) COMPLETE
        │           └─> Phase 5 (Training) COMPLETE
        │                 ├─> Phase 6 (Baselines)  PARTIAL (CLI + Colab ready)
        │                 └─> Phase 7 (Training)   PENDING (needs real data)
        │                       └─> Phase 8 (XAI)  PARTIAL
        │                             └─> Phase 10 (Paper) PARTIAL
        └─> Phase 4 (Data)    COMPLETE
                                    Phase 9 (Edge)  PENDING
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
| Source val saturates by epoch 1, best-val checkpoint uncorrelated with OOD quality | Early-stopping / model-selection picks wrong epoch for PlantDoc eval | Train fixed budget (15 epochs, patience=0), maintain weight EMA, save final EMA checkpoint; cell 45 auto-prefers final EMA when best-val was saved in epoch<2 | **Mitigated in notebook** -- `Copy_of_PhasePhyto_Colab.ipynb` cells 39/41/45 (v0.1.13) |
| 86M-param ViT backbone dominates training vs 150K-param PC encoder -> PC features never learn to discriminate | Physics stream contributes little to target F1, architecture reduces to "ViT with extra steps" | Differential LR (backbone at 10x lower LR) + auxiliary PC-only classifier head with aux CE loss weight 0.2 | **Mitigated in notebook** -- `Copy_of_PhasePhyto_Colab.ipynb` cells 34/39 (v0.1.13) |
| Weak source-side augmentation leaves training distribution far from PlantDoc -> source memorisation | Large source-target gap (-52% acc observed on 2026-04-17 run) | Strong aug stack: RandAugment + RandomPerspective + GaussianBlur + RandomRotation + stronger ColorJitter + shared-mask RandomErasing + HSV-masked background replacement | **Mitigated in notebook** -- `Copy_of_PhasePhyto_Colab.ipynb` cell 19 (v0.1.13) |
| SAM + AMP GradScaler interaction is fragile | Silent NaNs or divergent training when both are on | When `CONFIG["use_sam"]=True`, disable GradScaler; SAM runs two forward passes in FP32/autocast-off | **Mitigated in notebook** -- cell 39 gates `scaler` on `use_sam` (v0.1.13) |
| TENT mutates BN running stats on target batches -> cannot reuse the same model object for source eval afterwards | Source metric contaminated by target adaptation | Deep-copy model before TENT; evaluate source on original, target on adapted copy | **Mitigated in notebook** -- cell 45 deep-copies via `tent_adapt()` (v0.1.13) |
