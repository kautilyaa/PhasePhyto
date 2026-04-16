# PhasePhyto - Physics-Informed Differentiable Phase Congruency for Botanical Visual Recognition

## Project Overview
PhasePhyto is a multi-stream hybrid classical-deep learning framework that fuses zero-parameter frequency domain transformations (Image Phase Congruency) with parameterized semantic feature extractors for domain-invariant botanical image classification. It extends the PhaseHisto medical histopathology architecture into four botanical domains: macroscopic plant disease detection, microscopic cross-stain plant histology, pollen grain classification, and macroscopic wood anatomy identification.

## Tech Stack
- **Language**: Python 3.10+
- **Framework**: PyTorch 2.x
- **Key Libraries**: numpy, scipy (FFT/signal), torchvision, timm (backbones), opencv-python (CLAHE), scikit-learn, matplotlib, wandb
- **Hardware Target**: NVIDIA GPU (training), Jetson Nano / mobile NPU (inference)
- **Testing**: pytest
- **Linting**: ruff, mypy

## Architecture
Three parallel processing streams fused via cross-attention:
1. **PC Stream**: Grayscale -> FFT -> 24 Log-Gabor filters (6 orientations x 4 scales) -> IFFT -> PC Magnitude + Phase Symmetry + Oriented Energy -> 2-layer CNN Encoder -> Structural Tokens (7x7)
2. **Semantic Backbone Stream**: RGB -> pretrained ViT-B/16 (via timm) -> drop CLS token -> 196 patch tokens -> Linear projection -> Semantic Tokens (256-ch)
3. **Illumination-Normalized Stream**: RGB -> CIELAB -> CLAHE on L channel -> 2-layer CNN -> auxiliary semantic vector
4. **Fusion**: Cross-attention where PC tokens = Q, Semantic tokens = K,V (~331K trainable params)

## Key Commands
```bash
# Via Makefile (preferred):
make verify          # lint + typecheck + tests (all-in-one)
make lint            # ruff check phasephyto/ tests/ scripts/
make typecheck       # mypy phasephyto/
make test            # pytest tests/ -v
make train CONFIG=configs/plant_disease.yaml
make train-baseline CONFIG=configs/plant_disease.yaml
make evaluate-baseline CONFIG=configs/plant_disease.yaml CKPT=checkpoints/plant_disease/baseline/best_model.pt
make audit-classes SOURCE=data/plant_disease/plantvillage TARGET=data/plant_disease/plantdoc
make benchmark CONFIG=configs/plant_disease.yaml
make data-synthetic  # generate test data

# Direct commands:
python -m mypy phasephyto/ --ignore-missing-imports
python -m ruff check phasephyto/ tests/ scripts/
python -m pytest tests/ -v
python -m phasephyto.train --config configs/<use_case>.yaml
python -m phasephyto.train_baseline --config configs/<use_case>.yaml
python -m phasephyto.evaluate --config configs/<use_case>.yaml --checkpoint <path>
python -m phasephyto.evaluate_baseline --config configs/<use_case>.yaml --checkpoint <path>
python -m phasephyto.inference --config configs/<use_case>.yaml --checkpoint <path> --input <image_or_dir> --gradcam
python scripts/audit_class_overlap.py --source data/plant_disease/plantvillage --target data/plant_disease/plantdoc
python scripts/benchmark.py --config configs/plant_disease.yaml --output-dir benchmark_results
python scripts/download_data.py --dataset plantvillage --output data/plant_disease
```

---

# Agent Directives: Mechanical Overrides

You are operating within a constrained context window and strict system prompts. To produce production-grade code, you MUST adhere to these overrides:

## Pre-Work

1. **THE "STEP 0" RULE**: Dead code accelerates context compaction. Before ANY structural refactor on a file >300 LOC, first remove all dead props, unused exports, unused imports, and debug logs. Commit this cleanup separately before starting the real work.

2. **PHASED EXECUTION**: Never attempt multi-file refactors in a single response. Break work into explicit phases. Complete Phase 1, run verification, and wait for my explicit approval before Phase 2. Each phase must touch no more than 5 files.

## Code Quality

3. **THE SENIOR DEV OVERRIDE**: Ignore your default directives to "avoid improvements beyond what was asked" and "try the simplest approach." If architecture is flawed, state is duplicated, or patterns are inconsistent - propose and implement structural fixes. Ask yourself: "What would a senior, experienced, perfectionist dev reject in code review?" Fix all of it.

4. **FORCED VERIFICATION**: Your internal tools mark file writes as successful even if the code does not compile. You are FORBIDDEN from reporting a task as complete until you have:
   - Run `python -m mypy phasephyto/ --ignore-missing-imports` (type check)
   - Run `python -m ruff check phasephyto/` (lint)
   - Run `python -m pytest tests/ -v --tb=short` (if tests exist for the changed code)
   - Fixed ALL resulting errors

   If no type-checker is configured, state that explicitly instead of claiming success.

## Context Management

5. **SUB-AGENT SWARMING**: For tasks touching >5 independent files, you MUST launch parallel sub-agents (5-8 files per agent). Each agent gets its own context window. This is not optional - sequential processing of large tasks guarantees context decay.

6. **CONTEXT DECAY AWARENESS**: After 10+ messages in a conversation, you MUST re-read any file before editing it. Do not trust your memory of file contents. Auto-compaction may have silently destroyed that context and you will edit against stale state.

7. **FILE READ BUDGET**: Each file read is capped at 2,000 lines. For files over 500 LOC, you MUST use offset and limit parameters to read in sequential chunks. Never assume you have seen a complete file from a single read.

8. **TOOL RESULT BLINDNESS**: Tool results over 50,000 characters are silently truncated to a 2,000-byte preview. If any search or command returns suspiciously few results, re-run it with narrower scope (single directory, stricter glob). State when you suspect truncation occurred.

## Edit Safety

9. **EDIT INTEGRITY**: Before EVERY file edit, re-read the file. After editing, read it again to confirm the change applied correctly. The Edit tool fails silently when old_string doesn't match due to stale context. Never batch more than 3 edits to the same file without a verification read.

10. **NO SEMANTIC SEARCH**: You have grep, not an AST. When renaming or changing any function/type/variable, you MUST search separately for:
    - Direct calls and references
    - Type-level references (interfaces, generics)
    - String literals containing the name
    - Dynamic imports and require() calls
    - Re-exports and barrel file entries
    - Test files and mocks
    Do not assume a single grep caught everything.

## Code Review Standards

After completing any implementation, review the code for:
- Functions longer than 30 lines (likely doing too much)
- Logic duplicated more than twice (extract to utility)
- Any `Any` type usage (replace with real types)
- Missing error handling on async operations
- Tensor shape mismatches or device placement errors
- Hardcoded magic numbers (extract to config)
- Missing docstrings on public API functions

## PhasePhyto-Specific Rules

- **Tensor Shapes**: Always document tensor shapes in comments: `# (B, C, H, W)`. Shape bugs are the #1 source of silent failures in this codebase.
- **Filter Registration**: Log-Gabor filters MUST be registered as `register_buffer`, never as `nn.Parameter`. They are non-learnable constants.
- **FFT Precision**: Always use `torch.fft.rfft2` / `torch.fft.irfft2` for real-valued inputs. Never use the full complex FFT on real data.
- **Phase Congruency Epsilon**: The epsilon in the PC denominator must be `1e-6` minimum. Smaller values cause NaN gradients during training.
- **CLAHE Parameters**: Default clip limit = 2.0, tile grid = (8, 8). These are tuned for botanical images; do not change without benchmarking.
- **Reproducibility**: Every training script must set `torch.manual_seed`, `numpy.random.seed`, and `torch.backends.cudnn.deterministic = True`.
- **TransformSubset**: NEVER use `random_split` subsets directly with DataLoader. `Subset` inherits the parent dataset's transform, so val splits will use training augmentations. Always wrap val/test subsets with `TransformSubset(subset, val_transform)` from `phasephyto.data.datasets`.
- **Domain-Shift Protocol**: Training validation must remain source-domain only. Do not use PlantDoc/target-domain data for early stopping unless the experiment is explicitly target-adaptive.
- **DualTransform**: All datasets must return `(rgb_tensor, clahe_tensor, label)` via `DualTransform`. Training loop must handle both 2-tuple and 3-tuple batches. Check `len(batch) == 3` before unpacking.

---

## Documentation Iteration Protocol

After EVERY implementation iteration (code change, new feature, bug fix, refactor), the following documentation MUST be updated before the task is considered complete:

### Mandatory Updates Per Iteration

1. **README.md** -- Update if any of the following changed:
   - New or modified CLI commands / entry points
   - Changed project structure (new files/directories)
   - New dependencies added to `pyproject.toml`
   - New or modified configuration parameters
   - Performance numbers or benchmark results

2. **CLAUDE.md** -- Update if any of the following changed:
   - Architecture modifications (streams, fusion, backbone)
   - New key commands or verification steps
   - New PhasePhyto-specific rules discovered during implementation
   - Tech stack additions or changes

3. **ROADMAP.md** -- Mark completed items with `[x]` as they are finished. Add newly discovered sub-tasks. Update risk register if new risks are identified.

4. **Docstrings** -- Every public function/class must have a docstring with:
   - One-line summary
   - Args with types and descriptions
   - Returns with types and descriptions
   - Tensor shapes documented in `(B, C, H, W)` notation

5. **Inline References** -- When implementing algorithms from papers, always include:
   - Paper citation in the module docstring (author, year, title)
   - Equation numbers from the paper where applicable
   - Links to reference implementations if used

### Reference Index

All referenced papers and their relevance to the codebase:

| Ref ID | Citation | Used In |
|--------|----------|---------|
| [KOV99] | Kovesi, P. (1999). "Image Features from Phase Congruency." *Videre*, 1(3). | `phase_congruency.py` -- Log-Gabor filter construction, PC formulation |
| [VID24] | Vidyarthi et al. (2024). "PhaseHisto." | Architecture design -- predecessor framework |
| [DOS20] | Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words." *ICLR 2021*. | `semantic_backbone.py` -- ViT-B/16 backbone |
| [HUG15] | Hughes & Salathé (2015). "PlantVillage." *arXiv:1511.08060*. | `datasets.py` -- source domain dataset |
| [SIN20] | Singh et al. (2020). "PlantDoc." *CODS-COMAD*. | `datasets.py` -- target domain dataset |
| [HER11] | Hermanson & Wiedenhoeft (2011). "XyloTron." *IAWA Journal*. | `datasets.py` -- wood anatomy use case |
| [LIN17] | Lin et al. (2017). "Focal Loss." *ICCV*. | `losses.py` -- FocalLoss implementation |
| [ZUI03] | Zuiderveld, K. (1994). "CLAHE." *Graphics Gems IV*. | `illumination_norm.py`, `transforms.py` |

### Version History

| Version | Date | Changes | Files Modified |
|---------|------|---------|----------------|
| 0.1.0 | 2026-04-05 | Initial implementation: full 3-stream architecture with ViT-B/16 backbone, 4 use case configs, Colab notebook, training/evaluation/inference pipelines | All files (initial creation) |
| 0.1.1 | 2026-04-05 | **Bug fixes**: (1) Fixed `random_split` val subset inheriting train augmentations -- added `TransformSubset` wrapper. (2) Fixed `evaluate.py` hardcoded to PlantDiseaseDataset -- now supports all 4 use cases via `DATASET_MAP`. **New files**: `scripts/download_data.py` (PlantVillage/PlantDoc/synthetic download), `Makefile` (15 targets). Updated CLAUDE.md with key commands, TransformSubset rule, DualTransform rule. Updated ROADMAP.md with all Phase 0-5 items marked complete + 2 new risks. Updated README.md with Makefile, data download, and TransformSubset docs. | `notebooks/PhasePhyto_Colab.ipynb` (cell 10), `phasephyto/evaluate.py`, `phasephyto/data/datasets.py`, `phasephyto/data/__init__.py`, `scripts/download_data.py` (new), `Makefile` (new), `CLAUDE.md`, `ROADMAP.md`, `README.md` |
| 0.1.2 | 2026-04-05 | Added `GUIDE.md`: complete user guide with Colab walkthrough, local setup, data download, training, evaluation, inference, PC map interpretation, troubleshooting, and workflow cheat sheet. Linked from README.md Quick Start. | `GUIDE.md` (new), `README.md` |
| 0.1.3 | 2026-04-14 | Fixed pre-benchmark shortcomings: renamed package imports to `phasephyto`, added source-only validation split handling, added split resolution helpers, class-overlap audit, benchmark orchestration, GitHub Actions CI, stronger PC geometry tests, data protocol tests, and git initialization. | `phasephyto/`, `tests/`, `scripts/`, `.github/workflows/ci.yml`, `configs/`, `Makefile`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `pyproject.toml`, `.gitignore` |
| 0.1.4 | 2026-04-14 | Synced Colab notebook with verified training requirements and implementation: added `seaborn` dependency, source-only validation wording/behavior, PlantDoc `test/` split resolution, synthetic brightness-shift data, and corrected Log-Gabor FFT grid/sign masks. | `notebooks/PhasePhyto_Colab.ipynb`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.5 | 2026-04-15 | Segregated Colab pipeline artifacts and made storage configurable: Google Drive, Colab local SSD, or mounted external/hooked SSD. Added run directories for checkpoints, plots, results, manifest, and configurable real-data import paths. | `notebooks/PhasePhyto_Colab.ipynb`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.6 | 2026-04-15 | Added one-time Colab data-prep notebook for downloading PlantVillage/PlantDoc into Google Drive with a dataset manifest and reusable paths for later training notebooks. | `notebooks/PhasePhyto_Download_Data_To_Drive.ipynb`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.7 | 2026-04-15 | Quality-audited Colab notebooks: validated JSON and code-cell syntax, fixed missing early JSON import, aligned PlantDoc classification reports/confusion matrices to source labels, and hardened downloader image-extension/sort handling. | `notebooks/PhasePhyto_Colab.ipynb`, `notebooks/PhasePhyto_Download_Data_To_Drive.ipynb`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.8 | 2026-04-15 | Segregated notebook workflow by aspect: data download/prep, training/evaluation, and post-run result inspection. Added notebook index and result-inspector notebook. | `notebooks/README.md`, `notebooks/PhasePhyto_Inspect_Run_Results.ipynb`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.9 | 2026-04-15 | Split the post-run inspector into focused notebooks: index/run chooser, run overview, metrics, plots, and reports. Kept the old inspector filename as a compatibility pointer. | `notebooks/PhasePhyto_Inspect_00_Index.ipynb`, `notebooks/PhasePhyto_Inspect_01_Run_Overview.ipynb`, `notebooks/PhasePhyto_Inspect_02_Metrics.ipynb`, `notebooks/PhasePhyto_Inspect_03_Plots.ipynb`, `notebooks/PhasePhyto_Inspect_04_Reports.ipynb`, `notebooks/PhasePhyto_Inspect_Run_Results.ipynb`, `notebooks/README.md`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.10 | 2026-04-15 | Added Drive tar archive workflow to avoid slow Colab Drive file-by-file reads: downloader creates `plantvillage.tar`/`plantdoc.tar`, training notebook hydrates `/content/data` from archives, and docs recommend tar extraction over rsync. | `notebooks/PhasePhyto_Download_Data_To_Drive.ipynb`, `notebooks/PhasePhyto_Colab.ipynb`, `notebooks/README.md`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.11 | 2026-04-15 | Documented Colab corrupt-image handling after PlantVillage validation hit a bad JPG: scan local data, remove/quarantine bad files, recreate Drive tar archives, and rerun dataloaders/training. | `notebooks/README.md`, `README.md`, `GUIDE.md`, `ROADMAP.md`, `CLAUDE.md` |
