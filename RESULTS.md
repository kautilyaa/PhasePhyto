# PhasePhyto Results Log

Chronological record of training runs on the PlantVillage -> PlantDoc OOD
benchmark. Each entry captures what was changed, what was measured, and what
the analysis revealed. Runs are identified by their run directory under
`runs/<timestamp>_<ablation>/`.

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

1. Finish `backbone_only` ablation -- the single most important remaining
   number. Settles whether PC + fusion adds anything over a plain
   label-smoothed ViT under the same recipe.
2. Debug Phase 7.1.b pseudo-label (silently skipped in this run; threshold
   0.9 was almost certainly too tight for target calibration). Rerun with
   calibrated threshold, using the diagnostic confidence histogram to pick
   it.
3. Run `no_fusion` ablation to attribute cross-attention contribution.
4. One DANN attempt (Phase 7.1.c) as the last target-gradient lever. If it
   also fails to move target past ~55%, the negative result is complete
   and publishable as-is.
5. Optional: "best-target-snapshot" checkpoint alongside "best-val" in cell
   42 so the oracle-target number is always visible. Does not change the
   publishable metric but makes the OOD-selection pathology visible in
   every future run without requiring dedicated ablations.

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

| Date       | Run Dir                               | Ablation      | Source Acc | Source F1 | Target Acc | Target F1 | Delta Acc |
|------------|---------------------------------------|---------------|-----------:|----------:|-----------:|----------:|----------:|
| 2026-04-17 | (baseline, pre-OOD-hardening)         | full          | 0.9972     | 0.9948    | 0.4771     | 0.3603    | -0.5201   |
| 2026-04-20 | `20260420-181750_full`                | full          | 0.9970     | 0.9951    | 0.5033     | 0.3868    | -0.4937   |
| 2026-04-21 | `20260421-141232_pc_only`             | pc_only       | 0.7340     | 0.5878    | 0.0915     | 0.0791    | -0.6425   |
| 2026-04-21 | `20260421-202641_full` (v0.1.17)      | full+leafmask | 0.9975     | 0.9955    | 0.5229     | 0.3944    | -0.4746   |

Pending (Phase 7 ablation table — required for any PhasePhyto-vs-baseline claim):

| Planned    | Run Dir                               | Ablation          | Status                                  |
|------------|---------------------------------------|-------------------|-----------------------------------------|
| 2026-04-21 | `<ts>_backbone_only`                  | backbone_only     | IN PROGRESS (epoch 3/15: val_f1=0.9917) |
| TBD        | `<ts>_no_fusion`                      | no_fusion         | pending backbone_only                   |

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

## Run 2026-04-21 (in progress) -- `<ts>_backbone_only`

### Configuration

- Ablation: `backbone_only`. PC stream tokens zeroed out of the classifier
  head; model classifies from mean-pooled ViT patch tokens + illumination
  vector. `aux_pc_weight` auto-zeroed.
- Recipe identical to the 2026-04-20 `full` run.

### Observed so far

- Epoch 3 [SAM]: `train_loss=0.7390`, `train_acc=0.9652`, `val_loss=0.6377`,
  `val_acc=0.9959`, `val_f1=0.9917`.

### Interpretation (preliminary -- awaiting target numbers)

`backbone_only` is saturating source **faster** than the `full` run
(val_f1=0.9917 at epoch 3 vs. `full`'s 0.9951 at epoch 9). Whatever drag
the PC stream and cross-attention added to the optimization trajectory of
`full`, it was not contributing enough feature signal to justify the
slowdown. Plain ViT + the OOD-hardening recipe reaches near-peak source
validation in 3 epochs.

If the target number lands near `full`'s 0.5033, the full-vs-baseline
architectural claim collapses: the +2.6 target-acc gain `full` showed over
the 2026-04-17 pre-hardening baseline is attributable to the training
recipe (strong aug + bg replace + label smoothing + EMA + SAM + TENT), not
to the physics-informed fusion. The publishable story would shrink from
"physics-informed fusion for OOD leaf disease" to "strong augmentation
recipe for OOD on leaf disease."

### Next action when training finishes

1. Log source/target metrics in the Run Index above and in a full section
   below.
2. Decide whether to run `no_fusion` (only informative if `full` >
   `backbone_only`).
3. If `backbone_only` ≈ `full`, pivot to the Phase 7.1.a leaf-mask runs --
   that is the lever most likely to recover the PC stream's intended
   contribution on target.

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
