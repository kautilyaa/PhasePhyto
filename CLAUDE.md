# PhasePhyto - Rigorous OOD Study and Practical Recipe for Botanical Image Classification

## Project Overview

**Current framing (pivoted 2026-04-22, v0.1.18+).** PhasePhyto is a rigorous
empirical study of what does and does not transfer out-of-distribution for
botanical image classification, using PlantVillage (lab, clean) -> PlantDoc
(field, cluttered) as the benchmark. The codebase implements a multi-stream
architecture that fuses zero-parameter frequency domain transformations
(Image Phase Congruency, PC) with a ViT-B/16 semantic backbone, plus a
CLAHE-normalized illumination stream, via cross-attention.

**Headline empirical finding.** The PC stream, despite being mathematically
amplitude-invariant, **does not transfer OOD** on this benchmark. In the
`pc_only` ablation (2026-04-21) it scored target F1 = 0.08 against a random
baseline of 0.04 on 25 classes. The source signal (F1 = 0.59) does not
survive the distribution shift. The original thesis "physics-informed
fusion yields OOD-invariant features" is therefore not supported by the
data and is documented as a negative result.

**Practical contribution.** A controlled study of seven orthogonal
OOD-targeted levers (label smoothing, weight EMA, SAM, strong augmentation
+ background replacement, HSV leaf mask, hflip TTA, TENT test-time
adaptation) with per-lever target deltas and a complete ablation table
(`full`, `pc_only`, `backbone_only`, `no_fusion`). The practical recipe --
i.e., everything except the PC stream -- is reusable and directly
transferable to other botanical OOD benchmarks.

**Out-of-scope under the pivoted framing.** Expansion to use cases 2-4
(cross-stain histology, pollen grain, wood anatomy). The code for those
remains in the repo but no further experimental work is planned here.
Further architectural tweaks to the PC -> fusion pathway are deprioritized;
the ablations settle the question either way. See `RESULTS.md` for the
full pivot rationale and ROADMAP.md Phase 7.2 for the close-out plan.

### Four originally-planned botanical domains (now out of scope)

Kept here for code-archaeology reasons only: macroscopic plant disease
detection (the only active use case), microscopic cross-stain plant
histology, pollen grain classification, and macroscopic wood anatomy
identification. The `configs/` directory and dataset loaders still cover
all four; only use case 1 is being experimented on.

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
- **Phase Congruency Epsilon (split, do not collapse)**: Two different epsilons with opposite constraints; they must stay split. Collapsing them silently breaks the amplitude-invariance test.
  - `self.eps = 1e-6` goes only in the PC main denominator `(sum_A + self.eps)` and the phase-symmetry denominator. This is a true division-by-zero risk during training and must stay >= `1e-6` or gradients go NaN.
  - `self.sqrt_eps = 1e-12` goes inside every `torch.sqrt(... + sqrt_eps)` and in the non-degenerate denominators (min-max norm, width formula). It must be << `k**2 * min(even**2 + odd**2)` at the smallest k tested, otherwise `sqrt(even**2+odd**2+sqrt_eps)` floors the amplitude at a scale-independent constant and `PC(k*x) != PC(x)` for small k. **Do not write `self.sqrt_eps = math.sqrt(self.eps)`** -- that evaluates to `1e-3`, which is ~10^9 too large and makes every k fail the invariance test (verified 2026-04-21).
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

3. **RESULTS.md** -- The chronological run log. Update after EVERY completed
   training run (including ablations, failed runs, and negative results).
   See the "Memory and Results Logging Protocol" section below.

4. **ROADMAP.md** -- Mark completed items with `[x]` as they are finished. Add newly discovered sub-tasks. Update risk register if new risks are identified.

5. **Docstrings** -- Every public function/class must have a docstring with:
   - One-line summary
   - Args with types and descriptions
   - Returns with types and descriptions
   - Tensor shapes documented in `(B, C, H, W)` notation

6. **Inline References** -- When implementing algorithms from papers, always include:
   - Paper citation in the module docstring (author, year, title)
   - Equation numbers from the paper where applicable
   - Links to reference implementations if used

---

## Memory and Results Logging Protocol

Two complementary persistence layers are in use. They serve different purposes and both must be kept current.

| Layer | Location | Purpose | Audience |
|-------|----------|---------|----------|
| **`RESULTS.md`** | Repo root (checked in) | Authoritative, chronological, full-detail run log for every training experiment. | Anyone reading the repo, including future you. |
| **Auto-memory** | `~/.claude/projects/-Users-arunbhyashaswi-Drive-Code-PhasePhyto/memory/` | Compact, non-obvious context that the agent needs in future sessions. Pointers and implications, not raw data. | The agent across sessions. |

### `RESULTS.md` -- what goes in

Every training run -- ablation, config change, architectural experiment, even a failed or negative-result run -- MUST be recorded with:

1. A row in the **Run Index** table: date, run dir, ablation key, source acc, source F1, target acc, target F1, delta acc.
2. A full **Run entry** with: Configuration (all hyperparameters that differ from defaults), Metrics (table), Per-class breakdown (when available and informative), Analysis (what the numbers mean, what they disprove, what they confirm), Next steps (ordered by expected impact), Files (paths to artifacts).

Do not skip uninteresting runs. A negative result is often the most informative data point in the log.

### Auto-memory -- what goes in

Save a memory only when the content is:

- **Non-obvious** (not derivable by reading the current code or `RESULTS.md`)
- **Non-transient** (not current-task scratch state)
- **Cross-session relevant** (needed when a new conversation starts)

Specifically:

- **Project memories**: empirical state changes that reshape the project thesis (e.g., "the pc_only ablation proved PC is a regularizer, not an OOD feature source"). These should **point to `RESULTS.md` as the authoritative log** and capture only the *implication* that should shape future work.
- **Feedback memories**: preferences the user has validated (through correction OR through acceptance of a non-obvious framing). Lead with the rule, then `**Why:**` (the incident / confirmation) and `**How to apply:**` (when it kicks in).
- **Reference memories**: external-system pointers (Linear projects, Grafana boards, Drive paths) that the user references.

### Auto-memory -- what NOT to save

Do NOT save:

- Raw run numbers themselves -- they live in `RESULTS.md` and the memory should link there instead.
- Code snippets or fix recipes -- the fix is in the code; the commit message has the context.
- Cell numbers or file:line references -- these rot as the notebook and code evolve.
- Anything already documented in `CLAUDE.md` or `README.md`.
- Current task state, in-progress work, or ephemeral context.

### When to update

- **After every completed training run**: update `RESULTS.md` immediately. Then update the relevant project memory (`project_plantdoc_gap_state.md` or equivalent) if the run changed the empirical thesis of the project.
- **After user feedback reveals a new preference**: write a new feedback memory, or update an existing one if this refines prior guidance. Confirmations of non-obvious framings count as validations -- save from success as well as correction.
- **Before starting a new architectural direction**: re-read the relevant project memories and verify they still reflect current reality (check `RESULTS.md` and the code).

### Stale memory policy

Memory files display "this memory is N days old" when loaded. Treat any such memory as a point-in-time observation, not live state:

- Verify its claims against `RESULTS.md` and current code before acting.
- If stale, update the file rather than propagating outdated assumptions.
- If the fact is fully captured in `RESULTS.md` now, trim the memory to a pointer.

### Index hygiene

- `MEMORY.md` is always loaded into context; keep it under 20 lines / ~150 chars per entry.
- Each entry format: `- [Title](file.md) -- one-line hook that helps decide relevance`
- Do not write memory content directly into `MEMORY.md`; it is an index, not a memory.

---

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
| [FOR21] | Foret et al. (2021). "Sharpness-Aware Minimization for Efficiently Improving Generalization." *ICLR 2021*. | `notebooks/Copy_of_PhasePhyto_Colab.ipynb` cell 39 -- SAM optimizer |
| [WAN21] | Wang et al. (2021). "Tent: Fully Test-Time Adaptation by Entropy Minimization." *ICLR 2021*. | `notebooks/Copy_of_PhasePhyto_Colab.ipynb` cells 39/45 -- TENT adaptation |
| [CUB19] | Cubuk et al. (2019). "RandAugment: Practical Automated Data Augmentation." *NeurIPS 2020*. | `notebooks/Copy_of_PhasePhyto_Colab.ipynb` cell 19 -- training augmentation |
| [ZHO17] | Zhong et al. (2017). "Random Erasing Data Augmentation." *AAAI 2020*. | `notebooks/Copy_of_PhasePhyto_Colab.ipynb` cell 19 -- shared random erasing |

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
| 0.1.12 | 2026-04-16 | Added PlantDoc -> PlantVillage class alias mapping after manifest showed zero exact overlap. Audit reports mapped overlap, and Colab training creates a mapped target folder automatically when exact target samples are empty. | `phasephyto/data/class_mapping.py`, `scripts/audit_class_overlap.py`, `notebooks/PhasePhyto_Colab.ipynb`, `notebooks/PhasePhyto_Download_Data_To_Drive.ipynb`, `notebooks/RANCOPY_PhasePhyto_Colab.ipynb`, `tests/test_class_mapping.py`, `README.md`, `GUIDE.md`, `notebooks/README.md`, `ROADMAP.md`, `CLAUDE.md` |
| 0.1.14 | 2026-04-20 | Renamed `Copy_of_PhasePhyto_Colab.ipynb` back to the canonical `notebooks/PhasePhyto_Colab.ipynb` so the OOD-hardened notebook is the single training entry point (old file was deleted in git; no compatibility pointer needed). Added Phase 7.1 **ablation toggle** `CONFIG["ablation"]` in `{"full","pc_only","backbone_only","no_fusion"}`: architecture is unchanged across ablations; only the forward path branches -- `full` uses cross-attention, `pc_only` classifies from mean-pooled structural tokens + illum, `backbone_only` classifies from mean-pooled semantic tokens + illum, `no_fusion` averages the two mean-pooled streams without cross-attention. `aux_pc_weight` auto-zeroes in `backbone_only` since the PC stream is disabled. Run directories auto-suffix by ablation (`runs/<ts>_<ablation>/`) so four sequential runs produce four independent artifact trees. Sanity-check cell forces `ablation="full"` to keep PC-map shape assertions meaningful regardless of the run config. | `notebooks/PhasePhyto_Colab.ipynb` (renamed + cells 8, 10, 16, 34, 36), `CLAUDE.md`, `README.md` |
| 0.1.18 | 2026-04-22 | **Project thesis pivot: from "physics-informed fusion wins OOD" to "rigorous negative study + practical OOD recipe."** Rationale in `RESULTS.md` new "Project Thesis Pivot" section at top. Trigger: cumulative evidence from 2026-04-21 runs -- `pc_only` target F1 = 0.08 (random 0.04) + `full+leafmask` target acc 52.29% vs 2026-04-20 `full` target 50.33% (only +2.0 pts after all new levers) + target snapshots *dropping* across training (epoch 3 F1=0.3614 -> epoch 12 F1=0.3462) + Phase 7.1.b pseudo-label silently skipped (no `pseudo_phasephyto.pt` written, no `pseudo_*` keys in `history`). Combined, this disproves the original "PC + fusion yields OOD-invariant features" claim. **Documentation changes only, no code changes**: (a) `RESULTS.md` gets a pivot banner explaining why we moved off the physics claim and what still needs to close out the study. The 2026-04-21 `full+leafmask` run entry is added to the Run Index and as a detailed section. (b) `CLAUDE.md` Project Overview rewritten to the negative-study framing; four-use-case list moved to a footnote as "originally-planned, now out of scope." (c) `README.md` headline changes from "Physics-Informed Differentiable PC for Cross-Domain Botanical Visual Recognition" to "A Rigorous OOD Study and Practical Recipe for Botanical Image Classification." Adds results table, "What this project does and does NOT claim" section. (d) `ROADMAP.md` adds Phase 7.2 "Close-Out for Pivot" with five concrete items (finish backbone_only, debug pseudo-label, run no_fusion, one DANN attempt, optional best-target checkpoint, write-up pass) and moves use cases 2-4 to "DEPRIORITIZED under pivot." (e) Project memory `project_phasephyto.md` rewritten to the pivoted thesis with explicit "do not revive old framing" rule. (f) `MEMORY.md` index updated. | `RESULTS.md`, `CLAUDE.md`, `README.md`, `ROADMAP.md`, memory `project_phasephyto.md`, memory `project_plantdoc_gap_state.md`, memory `MEMORY.md` |
| 0.1.17 | 2026-04-21 | **Safety-net polish on Phase 7.1.a/b + amplitude-invariance regression fix.** (a) Fixed `self.sqrt_eps` in `PhaseCongruencyExtractor` (cell 26): was `math.sqrt(self.eps)` = 1e-3, which floored `sqrt(even**2+odd**2+sqrt_eps)` at a scale-independent constant and broke the `PC(k*x) == PC(x)` invariance test at every k. Restored to `1e-12` with an explanatory comment and an anti-regression rule added to the PhasePhyto-Specific Rules section of this file. Reproduction via numpy confirms fixed value passes all k at tiered tolerance; broken value fails with max_diff 0.45-1.0. (b) Pseudo-label phase (cell 43) now prints a target max-softmax decile histogram + counts above 0.5/0.7/0.8/0.9/0.95/0.99 BEFORE filtering, so threshold tuning is visible without rerun. (c) `aux_pc_weight` is hard-zeroed during the pseudo-label fine-tune (pc_only proved PC is decorative on OOD; spending fine-tune gradient on the aux PC pathway wastes budget). (d) Training loop (cell 42) now dumps full `history` dict to `RESULTS_DIR/history.json` so OOD trajectory survives kernel death. (e) All four `torch.save` calls (best, periodic, final-EMA, pseudo) now embed `"config": dict(CONFIG)` so a checkpoint can be re-attributed months later. (f) New sanity-viz cell at index 52 renders N source + N target images alongside their HSV foreground masks and prints foreground fraction per image -- a pre-flight check that the HSV threshold works on both datasets before a 15-epoch run. | `notebooks/PhasePhyto_Colab.ipynb` (cells 26, 42, 43, 52 (new)), `CLAUDE.md`, `README.md`, `ROADMAP.md` |
| 0.1.16 | 2026-04-21 | **Phase 7.1.b: pseudo-label self-training + Phase 7.1.a defaults flipped on.** Triggered by source saturating at val_f1 >= 0.99 for both `full` and `backbone_only` -- source-only regularisation cannot help further; only target-side gradient can. Added five new CONFIG knobs in `notebooks/PhasePhyto_Colab.ipynb` cell 8: `use_pseudo_label` (True in cell 10 override), `pseudo_label_threshold` (0.9), `pseudo_label_epochs` (5), `pseudo_label_lr_mult` (0.1 of base), `pseudo_label_min_samples` (50). Inserted a new code cell at index 43 (after the main training loop, before training curves) that: (1) uses the EMA model to predict on `target_loader` with no grad, (2) filters by confidence threshold, (3) builds a pseudo-labeled `PlantDiseaseDataset` with `train_tf` augmentation, (4) fine-tunes on `ConcatDataset(train_subset, pseudo_dataset)` for N epochs at reduced LR via a fresh AdamW optimiser and `train_epoch()` (SAM intentionally off during fine-tune), (5) runs optional per-epoch target snapshots into `history["pseudo_target_*"]`, (6) saves `pseudo_phasephyto.pt`. Extended cell 47 (was 46 pre-insertion) so the domain-shift evaluation prefers `pseudo_phasephyto.pt` over best-val and final-EMA when present. Also flipped Phase 7.1.a defaults **on** in cell 10: `leaf_mask_mode="hsv"`, `checkpoint_every=3`, `target_snapshot_every=3`. DANN / gradient reversal explicitly held as Phase 7.1.c last resort. `RESULTS.md` updated with four planned follow-up runs including two isolation runs to attribute gap-closure per lever; `ROADMAP.md` adds Phase 7.1.b and 7.1.c sub-tasks. | `notebooks/PhasePhyto_Colab.ipynb` (cells 8, 10, 43 (new), 47), `README.md`, `ROADMAP.md`, `RESULTS.md`, `CLAUDE.md` |
| 0.1.15 | 2026-04-21 | **Phase 7.1.a: OOD foreground segmentation + diagnostic hooks.** Triggered by the 2026-04-21 `pc_only` ablation (target F1=0.08) showing PC features do not transfer to PlantDoc -- most likely because PC is fitting PlantVillage background phase structure. Five new CONFIG knobs in `notebooks/PhasePhyto_Colab.ipynb` (all default off so prior runs reproduce): `leaf_mask_mode` in {`"off"`,`"hsv"`,`"hsv_blur"`}, `leaf_mask_sat_thresh` (default 40), `leaf_mask_blur_sigma` (default 1.5), `checkpoint_every` (0=off), `target_snapshot_every` (0=off). Added `_hsv_leaf_mask()` + `_apply_leaf_mask()` helpers and threaded them through `DualTransform` so the mask gates pixels before both CLAHE and PC on both train and val transforms (target eval inherits the gating). Added periodic-checkpoint save and target-domain snapshot inside the training loop so OOD trajectory is visible during training and the best-transferring epoch can be chosen post-hoc rather than trusting best-val-F1. Pseudo-labeling and DANN are explicitly deferred until the leaf-mask lever is measured. `RESULTS.md` updated with the planned `<ts>_full_leafmask` and `<ts>_full_leafmask_blur` runs; `ROADMAP.md` adds Phase 7.1.a sub-tasks under Use Case 1. | `notebooks/PhasePhyto_Colab.ipynb` (cells 8, 10, 19, 42), `README.md`, `ROADMAP.md`, `RESULTS.md`, `CLAUDE.md` |
| 0.1.13 | 2026-04-20 | **OOD-generalization hardening** for PlantVillage -> PlantDoc (prior run: source 99.72%, target 47.71%, gap -52%). Rewrote training notebook `Copy_of_PhasePhyto_Colab.ipynb` cells 8, 10, 19, 21, 34, 39, 41, 45, 46. Changes: (1) **Strong augmentation stack** with RandAugment, RandomPerspective, GaussianBlur, RandomRotation, stronger ColorJitter, shared-mask RandomErasing, and HSV-masked background replacement (solid-noise / gradient / high-freq styles) applied in PIL space before CLAHE so both streams see the same altered image. (2) **Label-smoothed CE** (0.1) replaces FocalLoss; FocalLoss left defined but unused. (3) **Differential LR**: ViT backbone at 10x lower LR than PC/fusion/head via `_build_param_groups`, with per-group-aware warmup. (4) **Weight EMA** (decay 0.999) used for validation and checkpointing; final EMA weights always saved separately; cell 45 auto-prefers final EMA when best-val was saved in epoch<2. (5) **Auxiliary PC-only classifier head** in `PhasePhyto` (aux_pc_head: LayerNorm->Linear->GELU->Dropout->Linear) with `return_aux` flag; training loop adds `aux_pc_weight` * CE(aux_logits, labels) to force PC stream to stay class-discriminative. (6) **SAM optimizer** (Foret et al. 2021) as opt-in class with two-step forward/backward; AMP disabled when SAM is active; separate `train_epoch_sam` function. (7) **TENT test-time adaptation** (Wang et al. 2021) on a deep-copy of the trained model: freezes all params except BN/LN affine (weight+bias), sets BN to train() to update running stats, minimizes Shannon entropy on target batches for N steps. (8) **hflip TTA** at target eval: softmax-averaged over image + horizontal flip. (9) **Class mapping audit**: cell 21 now resolves PlantDoc root across `test/`/`Test`/`val`/`valid`/base, adds case/underscore/punct-insensitive fallback lookup via normalized keys, and prints full diagnostic (mapped rows, skipped rows with reason, unknown folders not in dict, and source PV classes with zero target coverage). New CONFIG keys: label_smoothing, backbone_lr_mult, use_ema, ema_decay, use_tta, use_randaugment, randaugment_n, randaugment_m, random_erasing_p, aux_pc_weight, bg_replace_p, bg_saturation_thresh, use_sam, sam_rho, use_tent, tent_steps, tent_lr. Training budget reduced to 15 epochs, warmup 2, weight_decay 5e-2, dropout 0.2, head LR 3e-4 (prior best checkpoint fired at epoch 1 on a near-saturated source val). New paper refs added to Reference Index: [FOR21], [WAN21], [CUB19], [ZHO17]. | `notebooks/Copy_of_PhasePhyto_Colab.ipynb`, `CLAUDE.md`, `ROADMAP.md`, `README.md` |
