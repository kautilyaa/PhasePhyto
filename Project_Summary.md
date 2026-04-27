# PhasePhyto Project Summary

## One-line overview
PhasePhyto is a rigorous out-of-distribution (OOD) botanical image classification study on **PlantVillage -> PlantDoc**, with a key negative result and a reusable practical training recipe.

---

## Research goal
The original hypothesis was:

> Physics-informed phase congruency (PC) features fused with a ViT backbone would improve OOD transfer from lab images to field images.

The project has now pivoted to:

> A controlled empirical study of what does and does not transfer OOD, reporting negative findings honestly and extracting a practical recipe.

---

## What was built

- A 3-stream architecture:
  - **PC stream** (log-Gabor/phase congruency structural cues)
  - **ViT-B/16 semantic stream**
  - **CLAHE illumination-normalized stream**
- Cross-attention fusion between structural and semantic tokens.
- A strict ablation setup with shared training recipe:
  - `full`
  - `pc_only`
  - `backbone_only`
  - `no_fusion`

---

## Core findings

### 1) Main architectural claim is not supported
- `full` and `backbone_only` achieve essentially the same target accuracy (**0.5229**), with tiny F1 gap (~**0.009**) in likely noise range.
- This means PC+fusion does not show meaningful OOD advantage over a strong ViT path on this benchmark.

### 2) PC-only transfer fails OOD
- `pc_only` target F1 is very low (~**0.0791**, near random baseline for 25 classes).
- PC carries source signal, but it does not survive the domain shift adequately for classification.

### 3) Practical recipe is the transferable contribution
The OOD hardening stack (label smoothing, EMA, SAM, strong augmentation + background replacement, HSV leaf masking, TTA, TENT) improved target performance by about:

- **+4.6 accuracy points**
- **+3.4 macro-F1 points**

over the pre-hardening baseline.

---

## Novelty / contribution

1. **Rigorous negative-result paper-quality framing** instead of claim inflation.
2. **Controlled 4-cell ablation** under identical recipe for causal attribution.
3. Identification of the **invariance–classifier-head gap**:
   - Feature-level mathematical invariance does not guarantee classifier-level OOD invariance.
4. Practical, reproducible OOD training recipe for botanical tasks.
5. Strong reproducibility discipline (config snapshots, run logs, structured artifacts, tests, docs).

---

## Current performance snapshot (PlantVillage -> PlantDoc, mapped 25-class target)

| Run | Source Acc | Source F1 | Target Acc | Target F1 | Note |
|---|---:|---:|---:|---:|---|
| Baseline (pre-OOD hardening) | 99.72% | 0.9948 | 47.71% | 0.3603 | reference |
| `full` | 99.70% | 0.9951 | 50.33% | 0.3868 | improved |
| `pc_only` | 73.40% | 0.5878 | 9.15% | 0.0791 | fails OOD |
| `full+leafmask` | 99.75% | 0.9955 | 52.29% | 0.3944 | best full variant |
| `no_fusion` | 99.75% | 0.9955 | 49.02% | 0.3464 | below full |
| `backbone_only` | 99.75% | 0.9955 | 52.29% | 0.3859 | matches full acc |

---

## Strict apple-overlap (PV -> PlantDoc / PP2021, 3-class) -- new evidence (2026-04-26)

A second, narrower benchmark on the **shared 3-class apple label space**
(`Apple___healthy`, `Apple___Apple_scab`, `Apple___Cedar_apple_rust`) across
PlantVillage, PlantDoc, and Plant Pathology 2021. Trained on PV only with the
committed `configs/apple_overlap_plantdoc.yaml` recipe (ViT-B/16, AMP,
batch=32, focal loss, 30 epochs).

### Results (single seed, target was unseen during training)

| Target | n | Source Acc | Target Acc | Acc drop | Source F1 | Target F1 | F1 drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| PlantDoc test | 29 | 99.96% | 86.21% | -13.8 pp | 0.9996 | 0.8632 | -13.6 pp |
| Plant Pathology 2021 | 11,310 | 99.96% | 71.36% | -28.6 pp | 0.9996 | 0.6813 | -31.8 pp |

### What it shows

The PP2021 evaluation (n=11,310) is the statistically meaningful one. The
per-class drops are markedly asymmetric:

| Class | PP2021 n | Precision | Recall | F1 | Drop from PV F1 (=1.00) |
|---|---:|---:|---:|---:|---:|
| `Apple___healthy` | 4,624 | 0.69 | 0.94 | 0.80 | -20 pp |
| `Apple___Apple_scab` | 4,826 | 0.72 | 0.59 | 0.65 | -35 pp |
| `Apple___Cedar_apple_rust` | 1,860 | 0.81 | 0.47 | 0.60 | -40 pp |

Two distinct failure modes are visible in the confusion matrix:

1. **Healthy bias from PV class imbalance.** PV training distribution is
   ~65% healthy / 25% scab / 11% rust. Under the PP2021 target shift, the
   model defaults to "healthy" -- 1,771 of 4,826 actual scab cases (37%) are
   predicted as healthy, plus 174 of 1,860 rust cases (9%). This is the
   clinically dangerous failure mode: false negatives on disease.
2. **Rust collapses into scab.** 805 of 1,860 actual rust cases (43%) are
   predicted as scab. This is a *visual* failure, not a frequency failure --
   the model has learned PV's clean-background rust appearance and cannot
   re-identify the same disease under PP2021's orchard-context lighting.

### What this adds to the thesis

- **Independent corroboration of the negative-results headline at a different
  granularity.** Previously the PV-overstates-generalization story was carried
  by the 25-class PV->PD benchmark. The 3-class apple-overlap benchmark
  reproduces the pattern at a much smaller, fully-overlapping label space
  with a real-world n=11,310 target -- removing the "label mismatch is
  doing the work" alternative explanation.
- **Two-axis failure decomposition.** The PD->PP2021 gap difference (-14 pp
  vs -29 pp F1) and the rust->scab confusion together support framing the
  practical OOD recipe as *two interventions, not one*: a calibration /
  rebalancing axis for the healthy bias, and a feature-shift axis for the
  rust/scab confusion.
- **Concrete numbers for the write-up's "PV-only benchmarks are misleading"
  claim:** ~30 percentage points of accuracy on the same three apple labels.

### What this does not yet establish

- Single seed on this overlap; needs seeds 43, 44 to put error bars on the
  -29 pp number before publication.
- This is the **baseline** (essentially `backbone_only` in the apple
  context). It does not by itself say anything about whether PC+fusion
  helps or hurts on this overlap. The existing 25-class ablation shows full
  ~ backbone_only (+0.009 F1, noise); a backbone-only vs full re-run on the
  apple overlap would extend that finding to a tighter label space.
- PlantDoc-target numbers at n=29 carry a 95% CI of roughly +/-13 pp on
  accuracy. They are reported for completeness but should not be cited as
  evidence on their own.

### Follow-up interventions on PP2021 (2026-04-26): Fix A vs Fix B

Two follow-up attempts were run against the same PV-trained apple-overlap
baseline on PP2021:

- **Fix A (logit adjustment with uniform target prior):** net-negative.
- **Fix B (class-rebalanced retrain):** net-positive overall, but error
  redistribution across classes.

| Variant | Acc | F1 macro | Acc delta vs baseline | F1 delta vs baseline |
|---|---:|---:|---:|---:|
| Baseline (PV-trained) | 0.7136 | 0.6813 | -- | -- |
| Fix A (logit adjust, uniform prior) | 0.6789 | 0.6619 | -3.5 pp | -1.9 pp |
| Fix B (rebalanced retrain) | 0.7416 | 0.6969 | +2.8 pp | +1.6 pp |

Per-class F1 shows why aggregate metrics hide the mechanism:

| Class | Baseline | Fix A | Fix B | Best |
|---|---:|---:|---:|---|
| `Apple___Apple_scab` | 0.65 | 0.64 | 0.71 | Fix B (+5.8 pp) |
| `Apple___Cedar_apple_rust` | 0.60 | 0.58 | 0.56 | Baseline |
| `Apple___healthy` | 0.80 | 0.77 | 0.83 | Fix B (+3.2 pp) |

Interpretation:

1. **Fix A failed because the prior assumption was wrong.** The adjustment
   used `log(uniform) - log(PV_prior)`, but PP2021 is not uniform (~43% scab /
   16% rust / 41% healthy). This over-corrected rust and under-corrected scab,
   raising rust recall but collapsing rust precision and reducing healthy
   recall, with net macro-F1 decline.
2. **Fix B improved overall but hurt rust.** Weighted balancing with only 217 PV
   rust images oversampled rust too aggressively (roughly 6x exposure per
   epoch), likely memorizing PV rust appearance and reducing transfer to PP2021
   rust; scab and healthy improved while rust dropped.
3. **Neither intervention closes the domain gap.** Best target accuracy remains
   0.7416 versus ~1.00 on source, leaving a ~26 pp gap consistent with
   persistent feature shift.

**Fix B incidentally lifts PlantDoc too (n=29).**

| Variant on PlantDoc | Acc | F1 macro |
|---|---:|---:|
| Baseline (PV-trained) | 0.8621 | 0.8632 |
| Fix B (rebalanced retrain) | 0.8966 | 0.8965 |

Same caveat as the baseline PD result: n=29 carries a 95% CI of roughly
+/-13 pp, so this is directionally consistent with the PP2021 improvement
but should not be cited as standalone evidence.

This strengthens the thesis negative-result narrative: common calibration and
rebalancing fixes can move macro metrics while exposing non-trivial per-class
tradeoffs that matter more than aggregate gains.

Recommended low-cost follow-ups:

1. Re-run Fix A with `use_oracle_target_prior=True` (actual PP2021 prior) as an
   upper bound on prior-correction benefit.
2. Re-run Fix B with softer balancing weights (for example,
   `weight ∝ 1/sqrt(count)` via a `balanced_sampler_power` knob) to reduce
   overfitting pressure on the smallest class.

### Suggested next runs (lowest-cost first)

1. **Per-class threshold calibration on a held-out PP2021 slice** (no
   retrain). Quantifies how much of the -29 pp gap is calibration vs
   feature shift.
2. **PV class-balanced re-train** (1 retrain). If PP2021 rust/scab recall
   jumps, the failure was prior; if not, it is feature shift. Cleanest
   single experiment to disambiguate the two failure modes.
3. **Multi-seed (43, 44)** on the current setup for error bars.
4. **Backbone-only ablation on the apple overlap** to extend the existing
   25-class "full ~ backbone_only" result to the 3-class strict-overlap
   regime.

Raw eval JSONs and the combined summary are written by
`notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb` to:

```text
MyDrive/PhasePhyto/checkpoints/apple_overlap_plantdoc/
  eval_plantdoc.json
  eval_pp2021.json
  apple_overlap_eval_summary.{json,csv,md,png}
```

The chronological evidence log (per-run target classification reports,
domain-shift JSONs for `full` / `backbone_only` / `no_fusion`, apple-overlap
fix comparisons) is committed under `Results/` and synthesized in `RESULTS.md`.

---

## What the project now claims (and does not claim)

### Claims
- A reproducible OOD training recipe that improves transfer performance.
- A complete ablation-backed analysis of where gains come from.
- A valid negative finding about PC transfer on this benchmark.

### Does not claim
- That physics-informed fusion beats a ViT baseline under OOD.
- That PC features provide consistent discriminative OOD benefit in this setup.
- That conclusions are validated yet on other domains (histology/pollen/wood) in this study.

---

## Status

- Empirically in **close-out/write-up stage**.
- Major ablations completed (25-class PV->PD).
- 2026-04-26: strict 3-class apple-overlap baseline added as independent
  corroboration of the PV-overstates-generalization headline (PV->PP2021
  drop of -29 pp accuracy, -32 pp F1 on n=11,310; see section above).
- 2026-04-27: Fix A (logit adjustment) + Fix B (rebalanced retrain)
  evaluated on PP2021. Fix A net-negative (uniform-prior assumption wrong);
  Fix B net-positive (acc +2.8 pp, F1 +1.6 pp on PP2021; acc +3.4 pp on
  PlantDoc) but redistributes errors across classes. Full evidence under
  `Results/` and `RESULTS.md`.
- Remaining optional/last-mile items:
  - pseudo-label rerun with calibrated threshold (25-class)
  - optional DANN trial if needed (25-class)
  - apple-overlap multi-seed + Fix A oracle-prior re-run + softer Fix B
    sampler weighting (`balanced_sampler_power`) to tighten the per-class
    story before publication
  - final write-up polish

---

## Bottom line

PhasePhyto is a strong, honest OOD study: it disproves its initial headline hypothesis, explains why, and still delivers a practical recipe and reproducible methodology that others can build on.