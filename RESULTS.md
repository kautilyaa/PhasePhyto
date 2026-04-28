# PhasePhyto Results Log

Chronological record of training runs on the PlantVillage -> PlantDoc OOD
benchmark. Each entry captures what was changed, what was measured, and what
the analysis revealed. Runs are identified by their run directory under
`runs/<timestamp>_<ablation>/`.

---

## Research-Style Highlights (Results/ synthesis, 2026-04-27)

### Abstract

This study evaluates robust plant-disease transfer from controlled-source
training data (PlantVillage) to field-style target domains (PlantDoc and Plant
Pathology 2021), with emphasis on practical generalization improvements. Across
documented ablations and follow-up fixes (v1 + v2), the headline operational
outcome is that softened class rebalancing
(`balanced_sampler_power=0.5`) delivers the **best aggregate macro-F1 on
PP2021 (0.7393 acc / 0.7015 F1, n=11,310)** while matching the best
PlantDoc number (**0.8966 acc / 0.8965 F1, n=29**). The headline scientific
outcome is that the residual PV -> PP2021 gap is **feature-shift, not
prior**: even an oracle-prior post-hoc logit adjustment (Fix A v2)
underperforms the baseline by -0.3 pp macro-F1.

### Evidence Base (from checked-in `Results/`)

- Apple-overlap summary artifacts:
  `apple_overlap_plantdoc-20260427T160742Z-3-001/apple_overlap_plantdoc/apple_overlap_eval_summary.md`,
  `apple_overlap_plantdoc-20260427T160742Z-3-001/apple_overlap_plantdoc/apple_overlap_eval_summary.csv`,
  `apple_overlap_plantdoc-20260427T160742Z-3-001/apple_overlap_plantdoc/apple_overlap_eval_summary.png`
- Comparison tables:
  `apple_overlap_fixes_comparison-20260427T160744Z-3-001/apple_overlap_fixes_comparison/pp2021_macro_comparison.csv`,
  `apple_overlap_fixes_comparison-20260427T160744Z-3-001/apple_overlap_fixes_comparison/pp2021_per_class_comparison.csv`,
  `apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/pp2021_macro_comparison_v2.csv`,
  `apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/plantdoc_macro_comparison_v2.csv`,
  `apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/pp2021_per_class_comparison_v2.csv`
- 25-class ablation reports:
  `Fullresults-20260427T160555Z-3-001/results/target_classification_report.txt`,
  `Backboneresults-20260427T160446Z-3-001/results/target_classification_report.txt`,
  `NoFusionresults-20260427T160328Z-3-001/results/target_classification_report.txt`
- Plot evidence:
  `BackBoneplots-20260427T160505Z-3-001/plots/training_curves.png`,
  `NoFusionplots-20260427T160406Z-3-001/plots/training_curves.png`,
  `Fullplots-20260427T160557Z-3-001/plots/training_curves.png`,
  `Fullplots-20260427T160557Z-3-001/plots/confusion_matrices.png`,
  `Fullplots-20260427T160557Z-3-001/plots/illumination_invariance.png`,
  `Fullplots-20260427T160557Z-3-001/plots/leaf_mask_sanity.png`,
  `Fullplots-20260427T160557Z-3-001/plots/analysis_sample_0.png`,
  `Fullplots-20260427T160557Z-3-001/plots/analysis_sample_1.png`,
  `Fullplots-20260427T160557Z-3-001/plots/analysis_sample_2.png`

### Positive Findings

1. **Strong out-of-domain transfer on PlantDoc, further improved by softer
   rebalancing.**  
   Baseline apple-overlap reaches **0.8621 / 0.8632** (acc / macro-F1).
   Both Fix B variants (v1 full and v2 softer) lift PlantDoc to **0.8966 /
   0.8965** (+3.5 / +3.3 pp). PlantDoc-target n=29 is statistically
   anecdotal (95% CI ≈ ±13 pp) but is directionally consistent with the
   PP2021 result.

2. **Softer rebalancing is the new PP2021 macro-F1 headline.**  
   Fix B v2 (`balanced_sampler_power=0.5`, sqrt-softened inverse-frequency
   sampling) delivers **0.7393 acc / 0.7015 macro-F1** on PP2021, edging out
   Fix B v1 (full rebalance, 0.7416 acc / 0.6969 F1) on macro-F1 by +0.5 pp
   and recovering most of the rust regression Fix B v1 caused (rust F1
   −4.3 pp -> −1.6 pp vs baseline).

3. **PP2021 robustness from source-only training is meaningful.**  
   Baseline PV-trained overlap model achieves **0.7136 acc / 0.6813
   macro-F1** on the n=11,310 PP2021 target without any target-side tuning.

4. **Class-level gains on key apple categories under Fix B v2 (softer).**  
   - `Apple___Apple_scab`: **0.6506 -> 0.6981** F1 (+4.8 pp)
   - `Apple___healthy`: **0.7956 -> 0.8244** F1 (+2.9 pp)
   - `Apple___Cedar_apple_rust`: **0.5977 -> 0.5820** F1 (−1.6 pp;
     Fix B v1 was −4.3 pp on the same class -- softer cuts the regression
     by ~62%)

5. **Calibration-only fixes do not close the gap, even at oracle.**  
   Fix A v1 (uniform target prior): 0.6789 / 0.6619 (−3.5 / −1.9 pp).
   Fix A v2 (oracle PP2021 prior): 0.7049 / 0.6779 (−0.9 / −0.3 pp).
   Both are net-negative on macro-F1, isolating the residual gap as
   feature-shift rather than prior mis-specification.

6. **25-class ablation table is internally consistent across the three
   archived variants.**  
   `Fullresults` and `Backboneresults` both report **0.5229 target acc**
   (identical to four decimals) with target macro-F1 0.3944 / 0.3859
   respectively (Δ=+0.009 F1, inside run-to-run noise on n=153).
   `NoFusionresults` reports **0.4902 acc / 0.3464 F1**, confirming that
   inserting PC tokens without cross-attention gating actively hurts target
   transfer on this benchmark.

### Practical Research Contribution

- A reproducible, archive-first overlap workflow that supports rigorous
  cross-dataset benchmarking on a strict 3-class shared label space across
  PlantVillage, PlantDoc, and Plant Pathology 2021.
- A validated improvement path (`balanced_sampler_power=0.5`) that yields
  measurable gains on PP2021 macro-F1 without changing deployment
  assumptions, and recovers most of the rust-class regression that the
  full-rebalance variant caused.
- A clean negative isolation of post-hoc calibration: even with the oracle
  PP2021 prior, logit adjustment cannot recover the source-target gap, so
  follow-up effort should target feature shift (transductive TENT or a
  small target fine-tune) rather than further calibration.

### Conclusion

Taken together, the `Results/` artifacts support a measured research
narrative: the PhasePhyto pipeline already achieves solid OOD transfer on
the apple-overlap benchmark, and a sqrt-softened class-rebalanced retrain
(`balanced_sampler_power=0.5`) is the empirically-best single intervention
on PP2021 macro-F1. The remaining ~26 pp source-target accuracy gap is
not closable by further calibration or sampler tuning -- it is genuine
feature shift on the rust class, and is the right target for the next
intervention round (transductive adaptation or small target fine-tune).

---

## Final Study Summary (as of 2026-04-23, write-up stage)

**Study outcome.** The original PhasePhyto hypothesis -- that physics-informed
phase-congruency fusion beats a ViT baseline under OOD on botanical images --
is rejected by a complete four-cell ablation table under an identical
training recipe. `full+leafmask` (target F1 = 0.3944) and `backbone_only`
(target F1 = 0.3859) land at identical target accuracy (0.5229, four decimals)
with a +0.009 F1 delta inside run-to-run noise. `pc_only` at target F1 = 0.08
sits barely above the 25-class random baseline of 0.04, despite the PC
operator being FP32-verified amplitude-invariant. The transferable
contribution is the training recipe itself: +4.6 acc / +3.4 F1 over
pre-hardening baseline, identically present in `backbone_only`. See
**`PAPER.md`** for the draft manuscript.

**Four-cell ablation table, identical recipe, target with TTA+TENT:**

| Ablation        | Target Acc | Target F1 | Delta F1 vs full |
|-----------------|-----------:|----------:|-----------------:|
| `full+leafmask` |     0.5229 |    0.3944 | (ref)            |
| `backbone_only` |     0.5229 |    0.3859 | -0.0085 (noise)  |
| `no_fusion`     |     0.4902 |    0.3464 | -0.0480          |
| `pc_only`       |     0.0915 |    0.0791 | -0.3153          |

**Named failure mode.** The *invariance--classifier-head gap*: feature-level
mathematical invariance (PC(kx) == PC(x)) does not propagate through a
learned classifier head to OOD class-discriminative power, because classifier
weights fit source-specific statistics even of invariant features. See
`PAPER.md` §6.

**Remaining write-up-stage experiments (all optional, low priority):**
plain-timm-ViT row to close "your baseline is still your code" gap;
pseudo-label rerun at threshold 0.7 on `backbone_only`; DANN (declining).

---

## Project Thesis Pivot (2026-04-22)

**Old framing (v0.1.0 -- v0.1.17):** "Physics-informed differentiable phase
congruency, fused with a ViT backbone, yields domain-invariant features and
beats a ViT baseline under OOD on botanical images."

**New framing (v0.1.18+):** "A rigorous empirical study of what does and does
not transfer OOD for botanical image classification on PlantVillage ->
PlantDoc, with the PC-fusion result reported honestly as a negative finding
and a practical OOD training recipe offered as the reusable contribution."

**Why pivoted.** After three weeks of OOD-hardening across the `full`,
`pc_only`, and `full+leafmask` runs (see Run Index below), the data disproves
the original headline claim:

1. **`pc_only` target F1 = 0.08** (random = 0.04). The PC stream does not
   transfer OOD. Source F1 = 0.59 proves it carries signal on PlantVillage;
   target F1 = 0.08 proves that signal does not survive domain shift. The
   mathematical amplitude-invariance (`PC(kx) == PC(x)`, verified in cell 37)
   holds, but does not propagate through the classifier head to OOD
   class-discriminative power.
2. **Aggregate target movement in three weeks: +4.6 acc, +3.4 F1** (47.71 ->
   52.29 acc, 36.03 -> 39.44 F1). Across label smoothing, EMA, SAM, strong
   augmentation, background replacement, aux PC head, leaf mask, TENT, and
   hflip TTA stacked compound. Source sits saturated at 99.7% the entire
   time. The residual gap is distributional, not an optimization failure.
3. **Target snapshots *drop* across training** (this run: epoch 3 F1=0.3614
   -> epoch 12 F1=0.3462). The model's selected checkpoint (best source val)
   is *not* its best target checkpoint. Source-val-based selection itself is
   a pathology on this benchmark.

**What this pivot does NOT do.** It does not invalidate the work, the code,
or the architecture. All of those still ship. What changes is the claim we
make about them and the questions the write-up answers.

**Reusable contribution under the pivoted framing.** A controlled study of
seven orthogonal OOD-targeted levers (listed above), with per-lever target
deltas, an ablation table (`full` / `pc_only` / `backbone_only` / `no_fusion`),
and the documented failure modes. The practical recipe (everything except
the PC stream) is directly transferable to other botanical OOD benchmarks.

**Work still required to close out the study** (tracked in ROADMAP.md Phase 7.2):

1. [x] Finish `backbone_only` ablation (2026-04-23, target F1=0.3859).
   Settled the architectural question: `full` beats `backbone_only` by
   +0.009 target F1 and 0.000 target acc under the same recipe -- inside
   run-to-run noise, so PC + fusion does not beat plain ViT.
2. [x] Run `no_fusion` ablation (2026-04-23, target F1=0.3464). Cross-
   attention contributes +0.048 target F1 over mean-pool concat, but the
   `backbone_only` result then shows even cross-attention's gain does not
   survive vs removing the PC stream outright.
3. [ ] Rerun Phase 7.1.b pseudo-label with threshold=0.7 (diagnostic
   histogram from `no_fusion` run showed p95=0.904, so 0.9 is too tight
   for this model's target calibration; 0.7 would admit ~87 samples).
   Preferably run on top of `backbone_only` given its flatter target
   trajectory.
4. [ ] One DANN attempt (Phase 7.1.c) only if pseudo-label at 0.7 fails
   to push target F1 past ~0.42. With the ablation table closed, this
   attempt can be skipped if the recipe is judged sufficient.
5. [ ] Optional: "best-target-snapshot" checkpoint alongside "best-val"
   in cell 42 so the oracle-target number is always visible.
6. [ ] Write-up pass. Ablation table and per-lever deltas are now
   complete; the study is empirically closable.

**Explicitly deprioritized under the pivot:**

- Further architectural tweaks to the PC -> fusion pathway. `pc_only` +
  `backbone_only` settle the question either way; no new PC-stream design
  is worth building on this benchmark.
- Expansion to use cases 2-4 (wood anatomy, pollen, histology). These
  remain in the code but are out of scope for this thesis. If future work
  picks them up, the PC claim may recover in texture-dominated domains --
  but that is a separate study.
- Source-side regularization beyond what is already wired. Source is
  saturated; further source-only levers waste compute.

---

Unless noted otherwise, all runs use PlantVillage as source (lab, clean
backgrounds, uniform lighting) and PlantDoc as target (field, cluttered
backgrounds, variable lighting). Target evaluation uses the PlantDoc ->
PlantVillage class alias map from `phasephyto/data/class_mapping.py`.

---

## Run Index

### 25-class PV -> PlantDoc (n=153 target)

| Date       | Run Dir                               | Ablation      | Source Acc | Source F1 | Target Acc | Target F1 | Delta Acc |
|------------|---------------------------------------|---------------|-----------:|----------:|-----------:|----------:|----------:|
| 2026-04-17 | (baseline, pre-OOD-hardening)         | full          | 0.9972     | 0.9948    | 0.4771     | 0.3603    | -0.5201   |
| 2026-04-20 | `20260420-181750_full`                | full          | 0.9970     | 0.9951    | 0.5033     | 0.3868    | -0.4937   |
| 2026-04-21 | `20260421-141232_pc_only`             | pc_only       | 0.7340     | 0.5878    | 0.0915     | 0.0791    | -0.6425   |
| 2026-04-21 | `20260421-202641_full` (v0.1.17)      | full+leafmask | 0.9975     | 0.9955    | 0.5229     | 0.3944    | -0.4746   |
| 2026-04-23 | `20260423-132514_no_fusion`           | no_fusion     | 0.9975     | 0.9955    | 0.4902     | 0.3464    | -0.5073   |
| 2026-04-23 | `20260423-XXXXXX_backbone_only`       | backbone_only | 0.9975     | 0.9955    | 0.5229     | 0.3859    | -0.4746   |

**Ablation table complete as of 2026-04-23.** All four cells filled. The
architectural claim "PC + fusion beats plain ViT under OOD" is not supported
by this benchmark: `full` and `backbone_only` land at identical target
accuracy (0.5229) with only +0.009 target F1 advantage for `full`. See the
`backbone_only` entry below for the full analysis.

### Strict 3-class apple-overlap PV -> {PlantDoc, PP2021}

| Date       | Variant                                       | PD Acc | PD F1 | PP2021 Acc | PP2021 F1 | Notes |
|------------|-----------------------------------------------|-------:|------:|-----------:|----------:|---|
| 2026-04-26 | Baseline (PV-trained, single seed)            | 0.8621 | 0.8632 | 0.7136 | 0.6813 | reference |
| 2026-04-26 | Fix A v1 (`logit_adjust`, uniform prior)      |  n/a   |  n/a   | 0.6789 | 0.6619 | post-hoc, net-negative |
| 2026-04-26 | Fix B v1 (`balanced_sampler_power=1.0`)       | 0.8966 | 0.8965 | 0.7416 | 0.6969 | best PP2021 acc |
| 2026-04-27 | Fix A v2 (`logit_adjust`, oracle prior)       |  n/a   |  n/a   | 0.7049 | 0.6779 | upper bound; still net-negative |
| 2026-04-27 | **Fix B v2 (`balanced_sampler_power=0.5`)**   | **0.8966** | **0.8965** | **0.7393** | **0.7015** | **best PP2021 macro-F1 headline** |

PP2021 source numbers across all rows: 0.9996 / 0.9996 (acc / macro-F1) on
the PV apple-overlap source split; Fix B v1 hits 1.0000 / 1.0000 on its
own retraining split. The strict 3-class label space is too small to be a
meaningful ceiling differentiator; the interesting metric is the
target-side macro-F1.

Planned follow-up runs once the ablation table is complete (Phase 7.1.a +
7.1.b, added to `notebooks/PhasePhyto_Colab.ipynb` 2026-04-21):

| Planned    | Run Dir                               | Ablation      | Notable config delta                                                                                           |
|------------|---------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------|
| TBD        | `<ts>_full_leafmask_pseudo`           | full          | `leaf_mask_mode="hsv"`, `use_pseudo_label=True`, `checkpoint_every=3`, `target_snapshot_every=3`  (**v0.1.16 default**) |
| TBD        | `<ts>_full_leafmask_blur_pseudo`      | full          | `leaf_mask_mode="hsv_blur"`, `use_pseudo_label=True`                                                           |
| TBD        | `<ts>_full_leafmask_no_pseudo`        | full          | `leaf_mask_mode="hsv"`, `use_pseudo_label=False` (isolates leaf-mask contribution)                              |
| TBD        | `<ts>_full_pseudo_no_leafmask`        | full          | `leaf_mask_mode="off"`, `use_pseudo_label=True` (isolates pseudo-label contribution)                            |

Motivation and lever attribution:

- **Leaf mask (Phase 7.1.a)** — direct treatment for the 2026-04-21
  `pc_only` finding (target F1=0.08): PC fits PlantVillage background
  phase structure that PlantDoc does not share. Gate non-leaf pixels in
  `DualTransform` so PC cannot learn the background shortcut.
- **Pseudo-label self-training (Phase 7.1.b)** — adds target-side
  gradient. Both `full` and `backbone_only` saturate source val >= 0.99,
  so further source-only regularisation is wasted. Self-training one
  round on high-confidence target predictions (threshold=0.9, 5 epochs,
  lr/10) is the smallest lever that introduces target signal. DANN /
  gradient reversal is held as last resort if pseudo-label underperforms.

Isolation runs (rows 3 and 4) let us attribute gap-closure to each lever
individually. Skip them if row 1 already produces a meaningful lift.

---

## Run 2026-04-26 -- Apple-overlap PP2021 follow-ups (`fix_a_logit_adjust` vs `fix_b_rebalanced_retrain`)

### Context

Follow-up analysis on the strict 3-class apple-overlap baseline (PV-trained,
evaluated on PP2021) to test whether prior correction or class rebalancing
reduces the PV -> PP2021 transfer gap.

Compared variants:

- **Baseline:** original PV-trained overlap model.
- **Fix A:** logit adjustment using a **uniform** target prior.
- **Fix B:** retrain with class-rebalanced sampling.

### PP2021 metrics (n=11,310)

| Variant | Accuracy | F1 macro | Acc delta vs baseline | F1 delta vs baseline |
|---|---:|---:|---:|---:|
| Baseline (PV-trained) | 0.7136 | 0.6813 | -- | -- |
| Fix A (logit adjust, uniform prior) | 0.6789 | 0.6619 | -3.5 pp | -1.9 pp |
| Fix B (rebalanced retrain) | 0.7416 | 0.6969 | +2.8 pp | +1.6 pp |

### Per-class F1 (PP2021)

| Class | Baseline | Fix A | Fix B | Best |
|---|---:|---:|---:|---|
| `Apple___Apple_scab` | 0.65 | 0.64 | 0.71 | Fix B (+5.8 pp) |
| `Apple___Cedar_apple_rust` | 0.60 | 0.58 | 0.56 | Baseline |
| `Apple___healthy` | 0.80 | 0.77 | 0.83 | Fix B (+3.2 pp) |

### Analysis

**Fix A failed due to a wrong prior assumption.** The logit shift used a
uniform target prior, but PP2021 is not uniform (~43% scab / 16% rust /
41% healthy). That mismatch over-corrected rust and under-corrected scab,
improving rust recall but collapsing rust precision and reducing healthy recall;
net macro-F1 decreased.

**Fix B improved aggregate metrics but redistributed error.** Rebalancing helped
scab and healthy, but rust worsened. With only 217 rust images in PV, balanced
sampling overexposes the smallest class (roughly 6x per epoch), likely pushing
toward source-specific memorization rather than robust transfer.

**Domain gap remains substantial.** Even the best follow-up (Fix B) reaches
0.7416 accuracy on PP2021 versus near-perfect source metrics, leaving a large
residual gap consistent with feature shift beyond calibration/sampling alone.

### Low-cost next checks

1. Re-run Fix A with `use_oracle_target_prior=True` to measure the upper bound
   of prior-correction.
2. Re-run Fix B with softer sampler weighting (for example
   `weight ∝ 1/sqrt(count)`) via a `balanced_sampler_power` knob.

Both checks are now run; see the v2 follow-up section immediately below.

---

## Run 2026-04-27 -- Apple-overlap v2 follow-ups (`fix_a_oracle` + `fix_b_softer`)

### Context

The two "low-cost next checks" from the v1 follow-up section were specifically
designed as falsification tests for the v1 failure modes:

- **Fix A v1 failed** because uniform target prior mis-matched PP2021's
  ~43% scab / 16% rust / 41% healthy. Fix A v2 substitutes the **oracle**
  PP2021 prior (computed from the actual target labels), turning Fix A
  into an upper-bound measurement of how much post-hoc logit adjustment
  alone can recover.
- **Fix B v1 hurt rust** (-4.3 pp F1) because per-epoch oversampling of
  the small PV rust corpus (217 images) was ~3.13x. Fix B v2 introduces
  a `balanced_sampler_power` knob; setting `power=0.5` softens the
  inverse-frequency weighting to the square root, dropping rust
  oversampling to ~1.88x per epoch.

Both predictions are confirmed in the data.

### PP2021 macro metrics (n=11,310), 5-way comparison

From `Results/apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/pp2021_macro_comparison_v2.csv`:

| Variant | Acc | F1 macro | Precision macro | Recall macro |
|---|---:|---:|---:|---:|
| Baseline (PV-trained) | 0.7136 | 0.6813 | 0.7414 | 0.6676 |
| Fix A v1 (uniform prior) | 0.6789 | 0.6619 | 0.6543 | 0.6883 |
| Fix A v2 (oracle prior) | 0.7049 | 0.6779 | 0.7290 | 0.6572 |
| Fix B v1 (`power=1.0`) | 0.7416 | 0.6969 | 0.7495 | 0.6805 |
| **Fix B v2 (`power=0.5`)** | **0.7393** | **0.7015** | **0.7416** | **0.6873** |

Fix B v2 is the **best on macro-F1** (+0.5 pp over Fix B v1, +2.0 pp over
baseline). Fix B v1 retains a small accuracy edge (+0.2 pp), but the
F1 metric is the more honest summary on this 3-class shifted-prior target
because accuracy can be inflated by class collapse.

### PlantDoc macro metrics (n=29), 3-way comparison

From `Results/apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/plantdoc_macro_comparison_v2.csv`:

| Variant | Acc | F1 macro |
|---|---:|---:|
| Baseline | 0.8621 | 0.8632 |
| Fix B v1 (`power=1.0`) | **0.8966** | **0.8965** |
| Fix B v2 (`power=0.5`) | **0.8966** | **0.8965** |

Both rebalanced variants land at exactly the same PlantDoc point estimate
(`acc=0.8966, F1=0.8965`) -- reflecting that the dataset is too small
(n=29) to discriminate the two sampler powers and that the +3.5 / +3.3 pp
lift over baseline is recipe-attributable rather than power-attributable
at this granularity.

### PP2021 per-class F1, full 5-way comparison

From `Results/apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/pp2021_per_class_comparison_v2.csv`:

| Class | Baseline | Fix A v1 | Fix A v2 (oracle) | Fix B v1 (full) | Fix B v2 (softer) |
|---|---:|---:|---:|---:|---:|
| `Apple___Apple_scab` (n=4,826) | 0.6506 | 0.6357 | 0.6715 | **0.7085** | 0.6981 |
| `Apple___Cedar_apple_rust` (n=1,860) | **0.5977** | 0.5802 | 0.5870 | 0.5551 | 0.5820 |
| `Apple___healthy` (n=4,624) | 0.7956 | 0.7698 | 0.7751 | **0.8271** | 0.8244 |

Two readings:

1. **Rust regression is monotonic in sampler power.** Baseline (0.5977)
   > Fix B v2 softer (0.5820) > Fix B v1 full (0.5551). The dry-run
   sampler share matched the empirical outcome -- predictive
   correctness, not just numerical correctness.
2. **Scab is the class most responsive to rebalancing.** Baseline
   (0.6506) -> Fix B v2 softer (0.6981, +4.8 pp) -> Fix B v1 full
   (0.7085, +5.8 pp). Healthy moves +2.9 / +3.2 pp under v2 / v1
   respectively, with the same ordering.

### Source-side parity

Source (PV) accuracy and macro-F1 are unchanged across baseline, Fix B v1,
and Fix B v2 -- all three sit at 0.9996 / 0.9996 (Fix B v1 hits exactly
1.0000 / 1.0000 on its own train/eval split, but the 3-class source space
is essentially saturated regardless). Fix A is post-hoc logit adjustment
on top of the baseline checkpoint, so its source-side numbers degrade
slightly (0.9996 / 0.9991 with calibration applied, from
`eval_pp2021_calibrated.json`). The fixes do not bend source.

### Analysis

**Fix A is the smoking gun for "the gap is feature-shift, not prior."**
With the *oracle* PP2021 prior, the absolute upper bound of post-hoc
logit adjustment, macro-F1 still drops -0.3 pp vs baseline. The
deployable Fix A v1 (uniform prior) drops -1.9 pp. Calibration alone
cannot recover the source-target gap on this benchmark.

**`balanced_sampler_power` is a real, predictable knob.** Going from
`power=1.0` to `power=0.5` reduces rust oversampling from 3.13x to 1.88x
per epoch. The empirical rust F1 moved exactly in the predicted
direction (-4.3 pp -> -1.6 pp vs baseline). Future work can sweep
`power=0.25` cheaply to map the curve further toward the baseline.

**The domain gap is partially closable but not fully.** Best target
accuracy is now 0.7393 (Fix B v2 softer, best macro-F1) or 0.7416
(Fix B v1 full, best accuracy), versus near-perfect on source. ~26 pp
of source-target gap remains, consistent with genuine feature shift
that no recipe-side intervention has closed.

### Outcome summary

The v1 + v2 sweep produces a clean four-row narrative:

1. Uniform-prior calibration (Fix A v1) **fails**.
2. Oracle-prior calibration (Fix A v2) **also fails** -- isolates the
   residual gap as feature-shift.
3. Full rebalance (Fix B v1) **succeeds in aggregate, hurts rust**.
4. Softer rebalance (Fix B v2) **succeeds in aggregate AND mostly
   recovers rust** -- new headline.

### Files

Persisted under `Results/`:

- `apple_overlap_fixes_v2_comparison-20260427T181006Z-3-001/apple_overlap_fixes_v2_comparison/`
  -- 5-way macro and per-class comparison CSVs.
- `apple_overlap_plantdoc-20260427T160742Z-3-001/apple_overlap_plantdoc/`
  -- baseline/follow-up summary files:
  `apple_overlap_eval_summary.md`, `apple_overlap_eval_summary.csv`,
  `apple_overlap_eval_summary.png`.

Source notebook: `notebooks/PhasePhyto_Apple_Overlap_Fixes_v2_Colab.ipynb`.

### Optional remaining follow-ups (low cost)

1. `balanced_sampler_power=0.25` to see if the residual rust regression
   closes further. Diminishing returns expected; cheap to settle.
2. Multi-seed (43, 44) on Fix B v2 softer for error bars on the new
   headline F1 number.
3. Transductive next step: TENT on PP2021 or a small PP2021/PD
   fine-tune, targeting the residual rust feature-shift component
   specifically.

---

## Run 2026-04-23 -- `20260423-XXXXXX_backbone_only` (ablation: PC stream removed)

### Configuration

- Ablation: `backbone_only`. Forward path classifies from mean-pooled ViT
  semantic tokens + illumination vector only; PC stream tokens are not
  routed to the classifier head. `aux_pc_weight` auto-zeroed (PC stream
  disabled, so no aux loss).
- Recipe otherwise identical to the 2026-04-21 `full+leafmask` run (label
  smoothing 0.1, EMA 0.999, SAM rho=0.05, strong aug + bg replace, leaf
  mask `hsv`, TENT 20 steps, hflip TTA).
- Training budget: 15 epochs, warmup 2, base LR 3e-4. Checkpoint selected
  by eval: `best_phasephyto.pt` (val F1 epoch 13 = 0.9955).

### Metrics

| Metric      | Source (PlantVillage) | Target (PlantDoc, TTA+TENT) | Delta    |
|-------------|----------------------:|----------------------------:|---------:|
| Accuracy    | 0.9975                | 0.5229                      | -0.4746  |
| F1 macro    | 0.9955                | 0.3859                      | -0.6097  |

### Target snapshots across training (no TTA/TENT)

| Epoch | Target Acc | Target F1 |
|------:|-----------:|----------:|
| 3     | 0.4837     | 0.3785    |
| 6     | 0.5098     | 0.3961    |
| 9     | 0.4967     | 0.3625    |
| 12    | 0.5229     | 0.3942    |
| 15    | 0.5229     | 0.4027    |

**Drift pattern is different here.** Unlike every other full-stack run on
this benchmark, `backbone_only`'s target F1 trends *up* across training
(0.3785 at epoch 3 -> 0.4027 at epoch 15). The "best-val != best-target"
pathology is less severe for the pure-ViT path.

### Per-class target F1 (selected, from `target_classification_report.txt`)

Strong:
- Potato___Late_blight: F1=0.80 (n=8)
- Raspberry___healthy: F1=0.80 (n=7)
- Grape___healthy: F1=0.73 (n=12)
- Pepper,_bell___healthy: F1=0.71 (n=8)
- Soybean___healthy: F1=0.71 (n=8)
- Pepper,_bell___Bacterial_spot: F1=0.70 (n=9)
- Apple___Apple_scab: F1=0.67 (n=10)
- Apple___healthy: F1=0.60 (n=9)

Weak / collapsed:
- Cedar_apple_rust: F1=0.27 (n=10)
- Corn_Cercospora_leaf_spot: F1=0.31 (n=4)
- Corn_Common_rust: F1=0.31 (n=10)
- Peach___healthy: F1=0.36 (n=9, perfect precision, collapsed recall)
- Grape___Black_rot: F1=0.40 (n=8)
- Potato___Early_blight: F1=0.47 (n=8)
- Cherry___healthy: F1=0.40 (n=10)

### Analysis

**Pivoted thesis is now fully data-supported.** Against the 2026-04-21
`full+leafmask` run under an identical recipe:

| Ablation        | Target Acc | Target F1 | Delta vs full         |
|-----------------|-----------:|----------:|-----------------------|
| full+leafmask   | 0.5229     | 0.3944    | (reference)           |
| backbone_only   | 0.5229     | 0.3859    | 0.0000 acc, -0.0085 F1 |
| no_fusion       | 0.4902     | 0.3464    | -0.0327 acc, -0.0480 F1 |
| pc_only         | 0.0915     | 0.0791    | -0.4314 acc, -0.3153 F1 |

Target accuracy is **identical to four decimals** between `full` and
`backbone_only` (0.5229). The F1 delta is +0.009 in `full`'s favour --
plausibly within run-to-run noise given that the two runs share the same
checkpoint-selection pathology and differ only by the PC-stream and
cross-attention subgraph. The headline conclusion: **the PC stream and
cross-attention together earn at most +0.009 target F1 and 0.000 target
accuracy over a plain label-smoothed ViT under the same OOD recipe.**

**The no_fusion ordering is informative.** `backbone_only` (PC stream
absent) outperforms `no_fusion` (PC stream present but mean-pool
concatenated with ViT, no cross-attention) by +0.033 target acc / +0.040
target F1. When the PC stream is wired in without cross-attention routing,
it actively hurts target performance. Cross-attention's role appears to be
*suppressing* PC's background-phase noise on target, not adding
structural-token discriminative power. Removing the stream entirely is
cleaner than keeping it without gating.

**TTA+TENT was mildly harmful for backbone_only.** Epoch-15 target F1
without TTA/TENT = 0.4027; eval-reported F1 (with TTA+TENT) = 0.3859
(-0.017). Same direction as `no_fusion`. This suggests TENT's entropy
minimization is locking onto a wrong local minimum for the pure-ViT path
at this target size (n=153). Not critical for the headline claim, but
worth noting.

**Training dynamics confirm the earlier observation.** `backbone_only`
hits val_f1 >= 0.99 by epoch 2 and saturates by epoch 4 (0.9950). `full`
hit val_f1 = 0.9951 at epoch 9. The PC stream and fusion subgraph are an
optimization drag with no offsetting OOD benefit.

### Implication for the publishable claim

The pivoted thesis (v0.1.18, 2026-04-22) is now fully supported by the
data. Each of the four ablation cells contributes a concrete conclusion:

1. **pc_only (F1=0.08)**: PC features don't transfer OOD despite
   amplitude-invariance.
2. **no_fusion (F1=0.35)**: Inserting PC without cross-attention gating
   hurts target.
3. **backbone_only (F1=0.39)**: Plain ViT + OOD recipe matches `full`
   within noise.
4. **full (F1=0.39)**: The full stack provides 0.000 acc / +0.009 F1 over
   `backbone_only` -- i.e., nothing outside noise.

**The reusable contribution is the training recipe (label smoothing +
differential LR + EMA + SAM + strong aug + bg replace + leaf mask +
hflip TTA), not the physics-informed fusion.** Both `full+leafmask` and
`backbone_only` reach ~52.3% target accuracy under this recipe; the
pre-hardening baseline was 47.7%. The +4.6 acc gap is recipe-attributable.

### Next steps (priority order)

1. **Rerun pseudo-label at threshold=0.7** (still Phase 7.1.b). This is
   the last unexplored lever before DANN. Given `backbone_only`'s flatter
   target-snapshot trajectory, run pseudo-label on top of `backbone_only`
   rather than `full` -- less contaminated by fusion-stream regularization.
2. **Consider skipping DANN** (Phase 7.1.c). With the ablation table
   closed and the recipe identified, the only remaining question is
   whether target-side gradient can push target F1 past ~0.40 without
   the PC stream. One pseudo-label attempt is sufficient to settle it.
3. **Begin the write-up** (Phase 7.2 close-out). The ablation table and
   per-lever deltas are now a complete empirical picture. The
   PhasePhyto-vs-baseline claim is rescinded; the recipe becomes the
   reusable contribution. This is publishable as-is.

### Files

- Run dir: `/content/drive/MyDrive/PhasePhyto/runs/20260423-XXXXXX_backbone_only/`
- Checkpoints: `best_phasephyto.pt` (epoch 13, val_f1=0.9955),
  `final_ema_phasephyto.pt`. No `pseudo_phasephyto.pt` (pseudo-label
  config was not set for this run).
- Results: `results/history.json`, `results/phasephyto_domain_shift.json`,
  `results/target_classification_report.txt`.

---

## Run 2026-04-23 -- `20260423-132514_no_fusion` (ablation: cross-attention removed)

### Configuration

- Ablation: `no_fusion` (both streams present but fusion bypassed; final
  representation = mean-pooled structural tokens + mean-pooled semantic tokens
  concatenated, no cross-attention).
- Phase 7.1.a leaf mask on (`leaf_mask_mode="hsv"`), pseudo-label configured
  (`use_pseudo_label=True`, threshold=0.9, min_samples=50), diagnostic hooks on
  (`checkpoint_every=3`, `target_snapshot_every=3`). Otherwise identical to the
  2026-04-21 `full+leafmask` run.
- Training budget: 15 epochs, warmup 2, base LR 3e-4, SAM (rho=0.05, AMP off),
  EMA (0.999), label smoothing 0.1, `aux_pc_weight=0.2`, TENT (20 steps,
  lr=1e-3), hflip TTA.
- Checkpoint selected by eval: `best_phasephyto.pt` (val F1 epoch 13 = 0.9955).

### Metrics

| Metric      | Source (PlantVillage) | Target (PlantDoc, TTA+TENT) | Delta    |
|-------------|----------------------:|----------------------------:|---------:|
| Accuracy    | 0.9975                | 0.4902                      | -0.5073  |
| F1 macro    | 0.9955                | 0.3464                      | -0.6491  |

### Target snapshots across training (no TTA/TENT)

| Epoch | Target Acc | Target F1 |
|------:|-----------:|----------:|
| 3     | 0.5033     | 0.3892    |
| 6     | 0.4837     | 0.3767    |
| 9     | 0.4771     | 0.3476    |
| 12    | 0.4902     | 0.3488    |
| 15    | 0.4902     | 0.3383    |

### Pseudo-label phase

- Max-softmax deciles on target (p10/25/50/75/90/95/99): 0.349 / 0.512 / 0.812
  / 0.882 / 0.901 / 0.904 / 0.907.
- Counts at thresholds 0.5/0.7/0.8/0.9/0.95/0.99: 115 / 87 / 78 / 19 / 0 / 0.
- 19 confident samples at threshold 0.9, covering 10/25 classes. Below
  `pseudo_label_min_samples=50`, so pseudo-label fine-tune was skipped
  (diagnostic worked as intended). Top pseudo-classes: Raspberry___healthy (4),
  Apple___Apple_scab (3), Grape___healthy (3).

### Per-class target F1 (selected, from `target_classification_report.txt`)

Strong:
- Potato___Late_blight: F1=0.76 (n=8)
- Grape___healthy: F1=0.67 (n=12)
- Raspberry___healthy: F1=0.64 (n=7)
- Pepper,_bell___Bacterial_spot: F1=0.64 (n=9)
- Pepper,_bell___healthy: F1=0.62 (n=8)
- Apple___healthy: F1=0.62 (n=9)

Weak / collapsed:
- Cedar_apple_rust: F1=0.27 (n=10)
- Apple___Apple_scab: F1=0.54 (n=10)
- Corn_Cercospora_leaf_spot: F1=0.29 (n=4)
- Grape___Black_rot: F1=0.22 (n=8)

### Analysis

**Ablation-table delta: cross-attention buys ~0.048 target F1.** Against
the 2026-04-21 `full+leafmask` run (target F1 0.3944) under an identical
recipe, removing cross-attention in favour of mean-pooled concatenation
costs -0.048 target F1 and -0.033 target acc. Source is identical to
`full+leafmask` to four decimals (val_f1 = 0.9955 in both), so the cost
is purely on the target side. Interpretation: cross-attention contributes
modestly to OOD generalisation -- it helps structural tokens (PC) route to
the most relevant semantic-token neighbourhood -- but the contribution is
small relative to the overall ~0.40 source-target gap.

**Drift pattern matches prior runs.** Target snapshots drop monotonically
across training (epoch 3 F1=0.3892 -> epoch 15 F1=0.3383, -0.051). Same
selection pathology `full+leafmask` showed (epoch 3 F1=0.3614 -> epoch 15
F1=0.3591): source val continues to improve while target stalls or
degrades. Best-target-snapshot (epoch 3) outperforms best-val-selected
checkpoint (epoch 13). This further supports the 7.2 "optional best-target
checkpoint" item.

**Pseudo-label diagnostic worked.** Cell 43's histogram revealed that this
model's target confidence concentrates between 0.81 and 0.91 (p50 = 0.81,
p95 = 0.904), with essentially no samples above 0.95. At threshold 0.9
only 19/153 samples qualify, below the 50-sample floor, so the phase was
cleanly skipped with a logged reason -- exactly the instrumentation fix
the v0.1.17 patch was designed to produce. Next pseudo-label attempt
should lower the threshold to ~0.7 (would yield 87 samples, above the
floor) or move to a top-N-per-class selection.

**Implication for the ablation table.** With `no_fusion` now logged, three
of four ablation cells are filled:

| Ablation        | Target F1 (TTA+TENT) | Architecture present                         |
|-----------------|----------------------|----------------------------------------------|
| full+leafmask   | 0.3944               | PC + ViT + CLAHE + cross-attention           |
| no_fusion       | 0.3464               | PC + ViT + CLAHE, mean-pool concat instead   |
| pc_only         | 0.0791               | PC stream only (+CLAHE), no ViT              |
| backbone_only   | pending              | ViT + CLAHE only, no PC stream               |

`backbone_only` is still the load-bearing remaining number: it tells us
whether cross-attention and the PC stream together earn the architecture's
complexity over a plain label-smoothed ViT under the same recipe. With
`pc_only` at F1=0.08 and `no_fusion` at F1=0.35, the PC stream's marginal
contribution to OOD F1 could plausibly be negative in expectation.

### Next steps (priority order)

1. **Finish `backbone_only` run** (still pending). This is now the single
   outstanding experiment before the ablation table closes and the
   write-up can begin. If `backbone_only` >= 0.3464, the PC stream
   contributes nothing positive to OOD F1 and should be removed from the
   published recipe.
2. **Rerun pseudo-label with threshold=0.7** (would admit 87 target
   samples at current calibration). Still subject to `min_samples=50`
   guard. This is the one remaining source of target-side gradient before
   DANN.
3. **Consider DANN (Phase 7.1.c)** only if pseudo-label at 0.7 fails to
   lift target F1 above ~0.42. If DANN also fails, the negative result is
   publishable as-is.

### Files

- Run dir: `/content/drive/MyDrive/PhasePhyto/runs/20260423-132514_no_fusion/`
- Checkpoints: `best_phasephyto.pt` (epoch 13), `final_ema_phasephyto.pt`,
  periodic checkpoints at epochs 3/6/9/12/15 (no `pseudo_phasephyto.pt`
  because pseudo-label phase skipped).
- Results: `results/history.json`, `results/phasephyto_domain_shift.json`,
  `results/target_classification_report.txt`.

---

## Run 2026-04-21 -- `20260421-202641_full` (v0.1.17, leaf-mask on, pseudo-label configured)

### Configuration

- Ablation: `full` (PC + ViT + CLAHE + cross-attention).
- **Phase 7.1.a leaf mask on**: `leaf_mask_mode="hsv"`,
  `leaf_mask_sat_thresh=40`, `leaf_mask_blur_sigma=1.5`. HSV saturation
  foreground gate applied before CLAHE + PC on both train and val transforms.
- **Phase 7.1.b pseudo-label configured**: `use_pseudo_label=True`,
  `pseudo_label_threshold=0.9`, `pseudo_label_epochs=5`,
  `pseudo_label_lr_mult=0.1`, `pseudo_label_min_samples=50`.
- **Diagnostic hooks on**: `checkpoint_every=3`, `target_snapshot_every=3`.
- Otherwise identical to 2026-04-20 full recipe: strong aug + bg replace, label
  smoothing (0.1), differential LR (backbone_mult=0.1), EMA (0.999), SAM
  (rho=0.05, AMP off), `aux_pc_weight=0.2`, TENT (20 steps, lr=1e-3), hflip TTA.
- Training budget: 15 epochs, warmup 2, base LR 3e-4.
- Checkpoint selected by eval: `best_phasephyto.pt` (val F1 epoch 11 = 0.9955).

### Metrics

| Metric                      | Source (PlantVillage) | Target (PlantDoc, TTA+TENT) | Delta    |
|-----------------------------|----------------------:|----------------------------:|---------:|
| Accuracy (domain-shift JSON)| 0.9975                | 0.5229                      | -0.4746  |
| F1 macro (domain-shift JSON)| 0.9955                | 0.3944                      | -0.6011  |
| Accuracy (results JSON)     | 0.9975                | 0.5163                      | -0.4811  |
| F1 macro (results JSON)     | 0.9955                | 0.3979                      | -0.5976  |

Two numbers disagree slightly (0.5229 vs 0.5163) because
`phasephyto_domain_shift.json` is written after a second TENT pass, whereas
`phasephyto_results.json` is written during the main eval block. Treat the
domain-shift JSON as authoritative.

### Target snapshots across training (no TTA/TENT)

| Epoch | Target Acc | Target F1 |
|------:|-----------:|----------:|
| 3     | 0.4902     | 0.3614    |
| 6     | 0.4837     | 0.3590    |
| 9     | 0.4771     | 0.3559    |
| 12    | 0.4641     | 0.3462    |
| 15    | 0.4967     | 0.3591    |

### Per-class target F1 (selected, from `target_classification_report.txt`)

Strong:
- Pepper,_bell___Bacterial_spot: F1=0.80 (n=9)
- Grape___healthy: F1=0.76 (n=12)
- Raspberry___healthy: F1=0.74 (n=7)
- Potato___Late_blight: F1=0.74 (n=8)
- Apple___Apple_scab: F1=0.70 (n=10)

Weak:
- Cedar_apple_rust: F1=0.27 (n=10)
- Cherry___healthy: F1=0.33 (n=10)
- Grape___Black_rot: F1=0.31 (n=8)
- Corn___Cercospora_leaf_spot: F1=0.31 (n=4)

Zero-support (class in source but no PlantDoc samples map to it): Apple_Black_rot,
Cherry_Powdery_mildew, Corn_healthy, Grape_Esca, Grape_Leaf_blight_Isariopsis,
Orange_HLB, Peach_Bacterial_spot, Potato_healthy.

### Analysis

**Modest lift (+2.0 acc, +0.8 F1) over the 2026-04-20 `full` run.** Most of it
is attributable to the leaf mask + an extra TENT pass on a slightly different
checkpoint, not to Phase 7.1.b. Source is unchanged (already saturated).

**Target snapshots dropped across training (epoch 3 F1=0.3614 -> epoch 15
F1=0.3591)**, confirming the "best-val != best-target" pathology Phase 7.1.a
was supposed to diagnose. The leaf-mask gate changed the absolute level
slightly but did not fix the drift shape: the model still gets better at
source and simultaneously worse (or flat) at target as epochs progress. The
best target snapshot was epoch 3, before final convergence, which is exactly
the pattern `target_snapshot_every` was added to expose.

**Phase 7.1.b pseudo-label phase does not appear to have executed.** Evidence:

1. The eval JSON reports `"checkpoint": "best (val F1)"`. Cell 47 prefers
   `pseudo_phasephyto.pt` when present, so that file was not written.
2. `training_history` contains no `pseudo_target_acc` / `pseudo_target_f1`
   keys; only the 5 main-training snapshots.
3. No pseudo-label confidence histogram appears in the observed logs.

Most likely cause: either fewer than `pseudo_label_min_samples=50` target
samples cleared the 0.9 confidence threshold, or the phase silently failed
for another reason. The v0.1.17 confidence-histogram diagnostic was added
to reveal exactly this, so the fix is to rerun with the diagnostic cell in
place (already patched in cell 43) and inspect the printed deciles. If
p95 < 0.9 on target, the threshold is too tight for this model's target
calibration and should be loosened to e.g. 0.7 or switched to "top-N per
class" selection.

**Implication for the project thesis.** With pseudo-label apparently
skipped, the +2pt target gain vs 2026-04-20 came from leaf-mask + TTA/TENT
alone. That is consistent with the `pc_only` finding: source-side levers are
near exhausted, and any further target lift must come from actually routing
target-distribution gradient into the model. Until Phase 7.1.b actually
runs, DANN (Phase 7.1.c) remains deferred.

### Next steps (priority order)

1. **Re-run with pseudo-label diagnostics visible** (same config, but
   confirm cell 43 prints the max-softmax decile histogram and the
   min-samples check outcome). Expect the histogram to tell us whether the
   threshold is the blocker.
2. **If p95 target max-softmax < 0.9**, lower `pseudo_label_threshold` to
   0.7-0.8 and rerun. Alternatively switch to top-K-per-class selection so
   the floor is not threshold-sensitive.
3. **If pseudo-label runs and target F1 still stalls at ~0.40**, accept
   that source-only self-training is exhausted and move to Phase 7.1.c
   (DANN / gradient reversal) — the only remaining lever that forces the
   model to learn target-invariant features.
4. **Isolation runs** (previously queued) remain valuable but lower priority
   than (1): rerun `<ts>_full_leafmask_no_pseudo` and `<ts>_full_pseudo_no_leafmask`
   once pseudo-label is confirmed firing, to attribute gain per lever.

### Files

- Run artifacts: `/content/drive/MyDrive/PhasePhyto/runs/20260421-202641_full/`
- Domain-shift metrics JSON: `runs/20260421-202641_full/results/phasephyto_domain_shift.json`
- Results JSON: `runs/20260421-202641_full/results/phasephyto_results.json`
- Classification report: `runs/20260421-202641_full/results/target_classification_report.txt`
- Training curves: `runs/20260421-202641_full/plots/training_curves.png`
- Checkpoints: `best_phasephyto.pt` (epoch 11, val_f1=0.9955) and
  `final_ema_phasephyto.pt`. `pseudo_phasephyto.pt` NOT present -> confirms
  pseudo-label phase did not write a checkpoint.

---

## Run 2026-04-20 -- `20260420-181750_full`

### Configuration

- Ablation: `full` (PC stream + ViT stream + CLAHE stream + cross-attention fusion).
- Full OOD-hardening stack per README "Training Pipeline":
  - Strong paired augmentation (RandAugment, RandomPerspective, GaussianBlur,
    stronger ColorJitter, shared-mask RandomErasing, HSV-masked background
    replacement at `bg_replace_p=0.5`).
  - Label smoothing (eps=0.1). FocalLoss unused.
  - Differential LR (`backbone_lr_mult=0.1`).
  - Weight EMA (decay=0.999). EMA used for validation and checkpointing.
  - Auxiliary PC-only classifier head with `aux_pc_weight=0.2`.
  - SAM (Foret et al. 2021) enabled with `sam_rho=0.05`. AMP disabled.
  - TENT (Wang et al. 2021) at test time, `tent_steps=20`, `tent_lr=1e-3`.
  - hflip TTA at target evaluation.
- Training budget: 15 epochs, warmup 2, base LR 3e-4, weight decay 5e-2,
  dropout 0.2, PC orientations 8, num_heads 8.
- Checkpoint selection: best val F1 (`best_phasephyto.pt`), epoch 9,
  val_f1=0.9951.

### Metrics

| Metric        | Source (PlantVillage) | Target (PlantDoc) | Delta    |
|---------------|----------------------:|------------------:|---------:|
| Accuracy      | 0.9970                | 0.5033            | -0.4937  |
| F1 (macro)    | 0.9951                | 0.3868            | -0.6083  |

Target goal was `delta < 5%`. Achieved 49.4%.

### Delta vs. 2026-04-17 baseline

| Metric     | 2026-04-17 baseline | 2026-04-20 full OOD stack | Improvement |
|------------|--------------------:|--------------------------:|------------:|
| Source Acc | 0.9972              | 0.9970                    | -0.0002     |
| Source F1  | 0.9948              | 0.9951                    | +0.0003     |
| Target Acc | 0.4771              | 0.5033                    | +0.0262     |
| Target F1  | 0.3603              | 0.3868                    | +0.0265     |

### Analysis

**The stack worked, modestly.** The OOD-hardening stack produced +2.6 acc and
+2.7 F1 on target versus the pre-hardening baseline. This is real gain, but
far short of the target delta < 5% goal.

**Source is still saturated (99.70% / 99.51%).** All anti-memorization levers
(label smoothing, EMA, SAM, strong augmentation) did not prevent source
overfitting. The model continues to fit PlantVillage perfectly. When source
is saturated, the residual gap is almost certainly distributional (genuine
dataset shift) rather than an optimization pathology -- no amount of
additional source-side regularization is likely to help.

**The training process itself is healthier.** The prior baseline saved its
best val F1 at epoch 1, indicating the model memorized source before any
real learning happened. This run saved its best val F1 at epoch 9, meaning
the optimization curve is well-shaped and EMA smoothing is doing its job.
This is real structural progress even though target did not move much.

**What we still cannot tell without the ablation table:**

1. Is the PC stream class-discriminative or a glorified regularizer? The
   auxiliary PC-only head loss tells us during training, but we need a
   `pc_only` ablation run to see its standalone target performance.
2. Is the fusion doing work over a plain ViT? A `backbone_only` ablation
   would tell us whether the PC stream is adding anything on top of a
   label-smoothed, strongly-augmented ViT.
3. Is cross-attention better than averaging? A `no_fusion` ablation
   isolates this.

Without those three numbers, we cannot publish a PhasePhyto-vs-baseline
claim.

### Next steps (in priority order)

1. **Run `backbone_only` ablation immediately.** This is the single most
   informative number. If `backbone_only` target acc ~= 50%, then the entire
   PC + fusion stack is not adding value over a well-regularized ViT, and the
   architecture needs rethinking, not more training tricks. If
   `backbone_only` target acc is noticeably lower (say ~40%), then PC + fusion
   is contributing ~10 points and the complexity is justified.
2. **Run `pc_only` ablation.** If target is < 15%, the PC stream is not
   class-discriminative and functions only as a regularizer. If target is
   20-30%, PC carries meaningful structural information.
3. **Run `no_fusion` ablation.** Isolates the cross-attention contribution.
4. **Per-class target F1** (`PhasePhyto_Inspect_04_Reports.ipynb` on this
   run). If 5 classes are at 0% F1 and 10 are at 70%, the bulk of the gap
   is concentrated in a few classes -- likely class-mapping errors or
   severe class imbalance. That is a different, more tractable failure than
   a uniform distributional collapse.

Only after the three ablations and the per-class analysis should new levers
(e.g. stronger domain-adversarial training, segmentation-masked inputs,
target-side self-training) be considered. Diagnose first, then treat.

### Files

- Run artifacts:
  `/content/drive/MyDrive/PhasePhyto/runs/20260420-181750_full/`
- Domain-shift metrics JSON:
  `runs/20260420-181750_full/results/phasephyto_domain_shift.json`
- Checkpoints:
  `runs/20260420-181750_full/checkpoints/best_phasephyto.pt` (epoch 9,
  val_f1=0.9951) and `final_ema_phasephyto.pt` (end-of-training EMA weights).

---

## Run 2026-04-21 -- `20260421-141232_pc_only`

### Configuration

- Ablation: `pc_only`. Forward path classifies from mean-pooled structural
  tokens + illumination vector only; ViT semantic tokens are not routed to
  the classifier head. Architecture is unchanged; cross-attention is bypassed.
- Otherwise identical recipe to the `full` run: strong augmentation + bg
  replacement, label smoothing (eps=0.1), differential LR (backbone_mult=0.1,
  moot here), EMA (decay=0.999), SAM (`sam_rho=0.05`, AMP disabled), TENT at
  test time (`tent_steps=20`), hflip TTA.
- Training budget: 15 epochs, warmup 2, base LR 3e-4. Checkpoint selection:
  best val F1 at epoch 12 (val_f1=0.5878).

### Metrics

| Metric        | Source (PlantVillage) | Target (PlantDoc) | Delta    |
|---------------|----------------------:|------------------:|---------:|
| Accuracy      | 0.7340                | 0.0915            | -0.6425  |
| F1 (macro)    | 0.5878                | 0.0791            | -0.5087  |

Target goal was `delta < 5%`. Achieved 64.3%. Much worse than the `full` run,
as expected given the PC stream's capacity (~150K params vs. 86M in ViT).

### Per-class target F1 (selected)

Per the classification report in
`runs/20260421-141232_pc_only/results/target_classification_report.txt`, most
target classes are at F1=0. A small set has non-trivial F1:

- Potato___Early_blight: F1=0.40 (p=0.43, r=0.38, n=8)
- Peach___healthy: F1=0.36 (p=1.00, r=0.22, n=9)
- Pepper,_bell___healthy: F1=0.32 (p=0.27, r=0.38, n=8)
- Corn___Cercospora_leaf_spot: F1=0.25 (p=0.25, r=0.25, n=4)
- Blueberry___healthy: F1=0.18 (p=0.14, r=0.27, n=11)
- Apple___healthy: F1=0.15 (p=0.25, r=0.11, n=9)
- Corn___Common_rust: F1=0.15 (p=0.33, r=0.10, n=10)

Every other class is at F1=0 with support of 7-12 samples each.

### Analysis

**PC stream alone is class-discriminative on source but not on target.**
Source F1=0.59 is well above the 25-class random baseline (0.04), so phase
structure does carry discriminative information for some disease patterns
under PlantVillage's clean imaging. But target F1=0.08 is essentially random
— the PC features learned on PlantVillage do not transfer to PlantDoc's
cluttered, variable-lighting regime.

**This is a surprising negative result given the amplitude-invariance proof.**
Cell 37 verifies `PC(kx) == PC(x)` to within FP32 precision at
k in [0.5, 10]. That means the PC maps are scale-invariant in the raw-feature
sense. But the *learned classifier on top of those maps* over-fits the
PlantVillage texture statistics. The physical invariance guarantee does not
propagate through the classifier head.

**The classes that do transfer are instructive.** Peach__healthy,
Potato__Early_blight, and Corn__Cercospora all have strong directional or
periodic texture patterns (leaf veins, lesion striations, concentric rings)
that Log-Gabor filters are particularly well-suited to capture. The classes
that collapse tend to be visual categories defined more by color/context
than by oriented texture (healthy vs diseased often differs more in hue
than in edge structure when looked at through phase congruency).

**What this means for the `full` run's +2.6 acc gain.** It almost certainly
did NOT come from PC tokens contributing discriminative features to the
fusion. Target F1 = 0.08 for pc_only means the PC stream's output is not a
useful class signal at target. The aux PC head (weight 0.2) acted as a
**regularizer**, forcing gradients through the structural tokens, not as a
feature pathway. The actual target-side lift from `full` over the prior
baseline came from the training recipe: strong augmentation + background
replacement + label smoothing + EMA + SAM + TENT. The physics stream, as
currently wired, is decorative.

### Implications for the remaining ablations

`backbone_only` is now the critical run:

- If `backbone_only` target ~= 0.50, then PhasePhyto is not beating a
  well-regularized ViT. The publishable story reduces from "physics-informed
  fusion" to "strong augmentation recipe for domain shift on leaf disease."
- If `backbone_only` target is noticeably lower (say ~0.40), then the PC +
  fusion stack is contributing ~10 points through regularization rather than
  through feature content, and the architecture is still defensible.

`no_fusion` will tell us whether mean-pooled-then-averaged streams match
cross-attention; if yes, the cross-attention component is also decorative.

### Next step

Run `backbone_only` immediately. Do not run `no_fusion` yet — if
`backbone_only` matches `full`, the remaining ablation is less urgent and
the focus should shift to either (a) re-designing the PC -> fusion pathway
so PC features are preserved more strongly, (b) adding a foreground
segmentation step so PC sees only leaf pixels (background phase structure
is harmful noise on PlantDoc), or (c) reporting the strong-augmentation
recipe as the contribution and dropping the PC claim.

### Files

- Run artifacts:
  `/content/drive/MyDrive/PhasePhyto/runs/20260421-141232_pc_only/`
- Domain-shift metrics JSON:
  `runs/20260421-141232_pc_only/results/phasephyto_domain_shift.json`
- Classification report:
  `runs/20260421-141232_pc_only/results/target_classification_report.txt`
- Checkpoints: `best_phasephyto.pt` (epoch 12, val_f1=0.5878) and
  `final_ema_phasephyto.pt`.
