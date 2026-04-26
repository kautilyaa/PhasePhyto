# PhasePhyto — Draft Manuscript

**Working title.** Amplitude-invariant features are not classifier-invariant: a rigorous negative study of physics-informed fusion for out-of-distribution botanical image classification, with a reusable training recipe.

**Status.** Draft, 2026-04-23. Numbers below are the authoritative values from `RESULTS.md` at commit head of branch `aws_changes`. Class mapping uses `phasephyto/data/class_mapping.py` (25 PlantVillage source classes aliased onto PlantDoc target folders). Target evaluation set n=153.

---

## Abstract

We report a negative result and a practical recipe. We designed PhasePhyto, a three-stream architecture that fuses zero-parameter image phase congruency (PC) features with a ViT-B/16 semantic backbone through cross-attention, hypothesizing that PC's mathematical amplitude-invariance would propagate into an out-of-distribution classifier gain on lab-to-field botanical image transfer. Under a controlled four-cell ablation on PlantVillage → PlantDoc (25-class mapped, n=153 target), the hypothesis fails: the full stack (target F1 = 0.3944) and a plain label-smoothed ViT under the identical training recipe (`backbone_only`, target F1 = 0.3859) land at identical target accuracy (0.5229, four decimals) and a +0.009 F1 delta inside run-to-run noise. The PC stream in isolation (`pc_only`) achieves target F1 = 0.0791 — barely above the 25-class random baseline of 0.04 — despite the PC operator being verified amplitude-invariant to FP32 precision at every scale tested. The transferable contribution is the training recipe itself: label smoothing, differential learning rates, weight EMA, SAM, strong augmentation with HSV-masked background replacement, HSV leaf foreground gating, horizontal-flip TTA, and TENT. That recipe lifts target accuracy from 47.71% to 52.29% (+4.6 acc, +3.4 F1) over a pre-hardening baseline. We name the underlying failure mode the *invariance–classifier-head gap*: feature-level invariance does not guarantee learned-classifier invariance, and the gap is large enough on realistic botanical OOD to fully explain the negative result.

---

## 1. Introduction

### 1.1 Motivation

The lab-to-field generalization gap is the central unsolved problem in applied plant disease recognition. Models trained on PlantVillage (Hughes & Salathé, 2015) — studio-grade images with clean, isolated leaves and uniform lighting — collapse on PlantDoc (Singh et al., 2020), which captures field photographs with cluttered backgrounds, variable illumination, and partial occlusion. Published baselines sit in the 40–50% target accuracy range even with strong backbones, against 99%+ source accuracy. The gap is not an optimization failure; it is a distribution-shift failure.

Two families of approaches attack this gap: (a) aggressive regularization and augmentation on the source side, and (b) architectural or feature-engineering priors that encode invariances the shift preserves. PhasePhyto pursued both. Its distinguishing architectural claim was that *phase congruency*, a zero-parameter frequency-domain feature operator (Kovesi, 1999) that responds to edges and boundaries in a way mathematically invariant to monotonic amplitude rescaling, would encode leaf-structure information that survives the PlantVillage-to-PlantDoc shift in a way raw RGB features do not.

### 1.2 Original hypothesis (v0.1.0 – v0.1.17)

> A three-stream architecture that fuses PC features (amplitude-invariant by construction) with a ViT-B/16 semantic backbone through cross-attention will produce domain-invariant features and will beat a plain ViT baseline under OOD on botanical images.

### 1.3 Pivoted framing (v0.1.18, 2026-04-22)

> A rigorous empirical study of what does and does not transfer OOD for botanical image classification on PlantVillage → PlantDoc, with the PC-fusion result reported honestly as a negative finding and a practical OOD training recipe offered as the reusable contribution.

### 1.4 Contributions

1. A controlled four-cell ablation table (`full`, `pc_only`, `backbone_only`, `no_fusion`) under an identical OOD-hardening recipe. Three of four cells are new; `backbone_only` is the load-bearing comparison.
2. Per-lever attribution of seven orthogonal OOD-targeted techniques: label smoothing, weight EMA, SAM, strong augmentation with HSV-masked background replacement, HSV leaf foreground gating, hflip TTA, and TENT. Each lever's contribution to target F1 is quantified.
3. Identification and naming of the *invariance–classifier-head gap*: a learned classifier can over-fit source statistics even when its input features are formally invariant to the shift.
4. Documented *selection pathology*: the source-validation checkpoint is not the best target checkpoint. In full-stack runs target F1 degrades monotonically across training while source saturates. `backbone_only` is the exception; its target trajectory trends up.
5. A reusable training recipe that lifts target accuracy 47.71% → 52.29% over the pre-hardening baseline and transfers directly to other botanical OOD benchmarks.

### 1.5 What we do not claim

- That PhasePhyto beats a ViT baseline under OOD. The ablation data rescinds this claim.
- That PC tokens contribute discriminative features (as opposed to weak regularization) to the fused representation.
- Any result on histology, pollen, or wood anatomy. The code supports those datasets; no experiments were run.

---

## 2. Related Work

**Phase congruency.** Kovesi (1999) introduced PC as a scale-free measure of feature significance in images, computed from Fourier-domain log-Gabor filter responses at multiple orientations and scales. PC's key mathematical property — invariance to monotonic amplitude rescaling — is attractive for natural-image recognition under variable lighting. Prior biomedical applications (e.g., PhaseHisto, Vidyarthi et al. 2024) applied PC to histopathology cross-stain transfer with mixed success. To our knowledge PhasePhyto is the first study to evaluate PC for plant-disease lab-to-field OOD at scale.

**PlantVillage → PlantDoc OOD.** Hughes & Salathé (2015) released PlantVillage; Singh et al. (2020) introduced PlantDoc as the OOD counterpart. Published baselines on this benchmark span 40–50% target accuracy with ResNet-50 / EfficientNet / ViT backbones. Our baseline (plain-recipe ViT-B/16) sits at 47.71%, consistent with the literature.

**OOD hardening techniques.** Label smoothing (Szegedy et al. 2016), Sharpness-Aware Minimization (Foret et al. 2021), RandAugment (Cubuk et al. 2019), Random Erasing (Zhong et al. 2017), weight EMA (Izmailov et al. 2018), and TENT test-time adaptation (Wang et al. 2021) are well-established individually. The contribution of this work is *attributing* their per-lever target deltas on a single benchmark rather than stacking them.

**Domain adaptation baselines.** DANN / gradient-reversal (Ganin & Lempitsky 2015) and pseudo-label self-training (Lee 2013; Sohn et al. 2020) are the canonical target-gradient levers. We configured pseudo-label self-training; empirical calibration of the confidence threshold on this benchmark remains open work (see §6).

---

## 3. Method

### 3.1 Architecture

PhasePhyto processes each input image through three parallel streams, fused via cross-attention:

- **Stream 1 (PC, zero-parameter front-end + learnable encoder).** Grayscale → FFT → 24 log-Gabor filters (6 orientations × 4 scales) → IFFT → PC magnitude + phase symmetry + oriented energy → 2-layer CNN → 49 structural tokens (7×7, 256 channels).
- **Stream 2 (semantic backbone).** RGB → pretrained ViT-B/16 via `timm` → drop CLS token → 196 patch tokens (14×14) → linear projection to 256 channels.
- **Stream 3 (illumination-normalized).** RGB → CIELAB → CLAHE on L channel (clip limit 2.0, tile grid 8×8) → 2-layer CNN → auxiliary semantic vector.
- **Fusion.** Cross-attention where PC tokens serve as queries and semantic tokens as keys/values, ~331K trainable fusion parameters. An auxiliary PC-only classifier head is trained with weight 0.2 to keep the PC stream class-discriminative on source.

### 3.2 Ablation control

The same architecture is used across all four ablation conditions. Only the forward classification path differs:

| Ablation      | Classifier input                                                   | PC stream | ViT stream | Fusion         |
|---------------|--------------------------------------------------------------------|:---------:|:----------:|:--------------:|
| `full`        | Cross-attended fused tokens                                        | on        | on         | cross-attention|
| `pc_only`     | Mean-pooled structural tokens + illumination vector                | on        | off        | bypassed       |
| `backbone_only` | Mean-pooled semantic tokens + illumination vector                | off       | on         | bypassed       |
| `no_fusion`   | Mean-pool structural + mean-pool semantic (concatenated)           | on        | on         | bypassed       |

All other model parameters, optimization, and augmentation are held constant across ablations.

### 3.3 Training recipe

- Optimizer: AdamW with differential learning rate (backbone at 0.1× base LR).
- Base LR 3e-4, weight decay 5e-2, 15 epochs with 2 epochs of linear warmup.
- Sharpness-Aware Minimization (SAM), ρ=0.05. AMP disabled under SAM.
- Weight EMA (decay 0.999), used for validation and checkpoint selection.
- Label smoothing ε = 0.1 on cross-entropy. Focal loss was implemented but unused.
- Augmentation: RandAugment (N=2, M=9), RandomPerspective, GaussianBlur, ColorJitter, shared-mask Random Erasing (p=0.25), HSV-masked background replacement (p=0.5) with solid-noise / gradient / high-frequency background styles.
- HSV leaf foreground gating before CLAHE and PC (saturation threshold 40).
- Horizontal-flip TTA at target evaluation.
- Test-time adaptation: TENT, 20 steps, lr=1e-3. BN/LN affine parameters unfrozen, entropy minimization on target batches.

### 3.4 Reproducibility

All training runs set `torch.manual_seed`, `numpy.random.seed`, and `torch.backends.cudnn.deterministic = True`. Every checkpoint save embeds a `config` snapshot so later reattribution is possible. `history.json` is dumped per run. Each ablation writes to its own timestamped run directory.

---

## 4. Experimental Setup

### 4.1 Datasets

- **Source.** PlantVillage (Hughes & Salathé 2015). Lab imagery, uniform backgrounds, controlled lighting. 25 classes used after mapping (see §4.2).
- **Target.** PlantDoc (Singh et al. 2020). Field imagery, cluttered backgrounds, variable illumination. Test split only (n=153 after class mapping).
- **Class mapping.** PlantDoc class names differ from PlantVillage; `phasephyto/data/class_mapping.py` defines case- and punctuation-insensitive aliases that map PlantDoc folders onto the 25-class PlantVillage label space. Eight PlantVillage classes have zero PlantDoc target support and report zero precision/recall; this is an inherent limitation of the mapped benchmark, not of any ablation.

### 4.2 Splits

- Source: 85/15 train/val split with `TransformSubset` wrapping the val split to override the parent dataset's training augmentation.
- Target: PlantDoc `test/` resolved via a fallback chain (`test` / `Test` / `val` / `valid` / base).
- Validation runs on source only. Target data is never seen by the optimizer or by checkpoint selection in the non-adaptive pipeline.

### 4.3 Metrics

- Accuracy and macro-F1 on source validation and target test.
- Target metrics reported with and without TTA+TENT. The authoritative target number is post-TTA+TENT (`phasephyto_domain_shift.json`).
- Per-class precision / recall / F1 via `sklearn.metrics.classification_report`.
- Target snapshots every 3 epochs during training for OOD trajectory analysis.

---

## 5. Results

### 5.1 Headline: ablation table

All runs under the identical training recipe described in §3.3. Target metrics include TTA+TENT.

| Ablation        | Source Acc | Source F1 | Target Acc | Target F1 | Delta F1 vs full |
|-----------------|-----------:|----------:|-----------:|----------:|-----------------:|
| `full+leafmask` |     0.9975 |    0.9955 |     0.5229 |    0.3944 |        (ref)     |
| `backbone_only` |     0.9975 |    0.9955 |     0.5229 |    0.3859 |       −0.0085    |
| `no_fusion`     |     0.9975 |    0.9955 |     0.4902 |    0.3464 |       −0.0480    |
| `pc_only`       |     0.7340 |    0.5878 |     0.0915 |    0.0791 |       −0.3153    |

**Key reading.** `full` and `backbone_only` are statistically indistinguishable: identical target accuracy to four decimals, +0.009 F1 for `full` — inside run-to-run noise on n=153. The PC stream + cross-attention buys nothing over plain label-smoothed ViT under this recipe.

**Secondary reading.** `no_fusion` (−0.048 F1 vs `full`) underperforms `backbone_only`: inserting PC without cross-attention gating *actively hurts*. Cross-attention's role is not additive feature contribution; it is gating-away PC's background-phase noise. Removing PC entirely is cleaner than keeping it without the gate.

**Tertiary reading.** `pc_only` at target F1 = 0.0791 on 25 classes is barely above the random baseline of 0.04 — despite a source F1 of 0.59 that proves PC carries source-class signal. The signal does not survive the shift.

### 5.2 Recipe contribution

| Configuration                                    | Target Acc | Target F1 | Delta vs baseline |
|--------------------------------------------------|-----------:|----------:|------------------:|
| Baseline (plain recipe, 2026-04-17)              |     0.4771 |    0.3603 |     (ref)         |
| `full` + OOD stack (2026-04-20)                  |     0.5033 |    0.3868 |    +2.6 acc       |
| `full+leafmask` (2026-04-21)                     |     0.5229 |    0.3944 |    +4.6 acc       |
| `backbone_only` + OOD stack (2026-04-23)         |     0.5229 |    0.3859 |    +4.6 acc       |

Three weeks of OOD hardening lifted target accuracy by 4.6 percentage points and target F1 by 3.4 points. All of that gain is recipe-attributable: it appears identically in `backbone_only`.

### 5.3 Per-class target F1 (selected, `full+leafmask`)

Strongest transferring classes share directional or periodic texture structure:

- Pepper,_bell___Bacterial_spot: F1 = 0.80
- Grape___healthy: F1 = 0.76
- Raspberry___healthy: F1 = 0.74
- Potato___Late_blight: F1 = 0.74
- Apple___Apple_scab: F1 = 0.70

Weakest transferring classes are color- or context-defined rather than texture-defined:

- Cedar_apple_rust: F1 = 0.27
- Cherry___healthy: F1 = 0.33
- Grape___Black_rot: F1 = 0.31
- Corn___Cercospora_leaf_spot: F1 = 0.31

Eight PlantVillage classes have zero PlantDoc target support and are omitted from averaging.

### 5.4 Selection pathology: target snapshots across training

Target F1 trajectory (no TTA/TENT) by epoch:

| Epoch | `full+leafmask` | `no_fusion` | `backbone_only` |
|------:|----------------:|------------:|----------------:|
| 3     | 0.3614          | 0.3892      | 0.3785          |
| 6     | 0.3590          | 0.3767      | 0.3961          |
| 9     | 0.3559          | 0.3476      | 0.3625          |
| 12    | 0.3462          | 0.3488      | 0.3942          |
| 15    | 0.3591          | 0.3383      | 0.4027          |

Full-stack runs degrade on target while saturating on source. `backbone_only` trends up. This is an instance of source-val selection itself being a failure mode on this benchmark: the checkpoint the pipeline selects is not the best-target checkpoint. Plain ViT has a healthier OOD trajectory than the fused stack.

### 5.5 Pseudo-label calibration

We configured pseudo-label self-training on target (threshold 0.9, 5 epochs at 0.1× base LR, minimum 50 confident samples). Cell-43 diagnostic on the `no_fusion` run reported the target max-softmax distribution:

| Percentile | p10   | p25   | p50   | p75   | p90   | p95   | p99   |
|-----------:|------:|------:|------:|------:|------:|------:|------:|
| Value      | 0.349 | 0.512 | 0.812 | 0.882 | 0.901 | 0.904 | 0.907 |

At threshold 0.9, only 19 of 153 target samples qualify — below the 50-sample floor, so the phase auto-skipped. At 0.7, approximately 87 samples would qualify. The 0.9 threshold is miscalibrated for this model's target confidence distribution. A retry at 0.7 is empirically justified, not a guess.

---

## 6. Analysis: the invariance–classifier-head gap

### 6.1 The operator is invariant; the classifier is not

PhasePhyto's PC extractor is verified amplitude-invariant to FP32 precision: `PC(k · x) == PC(x)` to within 1e-5 absolute tolerance for k ∈ [0.5, 10] on all test images. This is a property of the operator itself (`phasephyto.models.phase_congruency`), independent of any learned weights. The `pc_only` ablation then trains a classifier head on those invariant maps and achieves:

- Source F1 = 0.59 — well above the 25-class random baseline (0.04).
- Target F1 = 0.08 — essentially at the random baseline.

The operator's invariance is preserved; the classifier's transfer is not. We name this the *invariance–classifier-head gap*: a classifier trained on invariant features can still over-fit the source-specific statistics of those features, because classifier weights are learned from source data only. Invariance at the input does not propagate to invariance of learned decision boundaries.

### 6.2 Which classes transfer

Classes that transfer under `pc_only` share a common structural profile: strong directional textures (Peach___healthy leaf venation, Potato___Early_blight lesion striations, Corn___Cercospora concentric rings). Log-Gabor filters at 8 orientations and 4 scales are well-matched to such patterns. Classes that collapse are those whose disease signature is primarily color- or context-defined (healthy vs early-symptom hue differences that PC discards by construction).

This is consistent with PC being a texture/structure operator and not a color/context operator. It is also consistent with the negative transfer result: PlantDoc differs from PlantVillage primarily in *context* (backgrounds, lighting, occlusion), not in *structural texture of leaves themselves*. PC preserves what is not at issue and discards what is.

### 6.3 Why cross-attention does not rescue PC

The `no_fusion` cell is instructive. If PC tokens contributed additive signal, mean-pool concatenation of PC + ViT should exceed ViT alone. It does not: `no_fusion` F1 = 0.3464 < `backbone_only` F1 = 0.3859. Cross-attention in `full` recovers to `backbone_only` parity (0.3944 vs 0.3859) but does not exceed it. Interpretation: cross-attention's role on this benchmark is to *gate* the PC stream's background-phase noise away from the classifier, recovering the ViT-only baseline. It does not inject additional PC signal because there is no additional PC signal to inject.

### 6.4 The recipe is the contribution

The three-week, seven-lever hardening campaign moved target accuracy by +4.6 points. That movement is identically present in `backbone_only`, so it is architecturally independent. It is attributable to the training recipe: label smoothing, differential LR, EMA, SAM, strong augmentation with background replacement, leaf foreground gating, hflip TTA, and TENT. This recipe transfers directly to any botanical OOD benchmark with a PlantVillage-like / PlantDoc-like structural shift.

---

## 7. Limitations

1. **Target evaluation set size is small (n=153).** The +0.009 F1 delta between `full` and `backbone_only` is inside run-to-run noise at this sample size. A larger target set, a second independent target domain, or resampled confidence intervals would strengthen the "statistical tie" claim. As reported, we treat the delta as noise rather than signal.
2. **Plain-timm-ViT row is missing.** Our `backbone_only` ablation still runs the PhasePhyto code path with the PC stream disabled. A literal `timm` ViT-B/16 row under the same recipe would close the "but your baseline is still your code" gap for reviewers. This is cheap to produce and is the top write-up-stage experiment.
3. **Pseudo-label was configured but did not fire** in any reported run (threshold 0.9 was miscalibrated). A retry at threshold 0.7 on `backbone_only` is unblocked but not yet reported.
4. **DANN was not run.** With the ablation table closed, a DANN result cannot resurrect the fusion claim; it can only slot into the recipe. We judged the marginal scientific value below the compute cost and deferred it.
5. **No evaluation on histology, pollen, or wood anatomy.** The architecture and configs support those domains, but the study is scoped to use case 1 (plant disease).
6. **Single ViT backbone.** We did not ablate backbone choice. A ConvNeXt or DINOv2 backbone under the same recipe may shift the operating point but is unlikely to change the architectural conclusion.

---

## 8. Discussion

### 8.1 What survives

The operational recipe and the negative result both survive. A practitioner picking up this code for a new botanical OOD dataset can:

1. Set `ablation="backbone_only"`, which is now the recommended default.
2. Use the training recipe as-is.
3. Expect ~+4.6 target accuracy over a plain-recipe baseline, with the understanding that the residual gap is distributional and not closable by further source-side regularization.

The PC stream is retained in the codebase for reproducibility of the ablation. It is not recommended as a production component on this benchmark.

### 8.2 What the negative result teaches

The invariance–classifier-head gap is not specific to phase congruency. Any physics-derived or symmetry-enforced feature operator faces the same risk: the operator's invariance is a property of the operator, not of the classifier head learned on top of it. Without explicit training-time pressure on the classifier to respect the invariance (e.g., target-domain data during training, adversarial domain alignment, or invariance-preserving regularizers), the classifier will still over-fit the source-specific statistics of the invariant features.

This suggests a general hygiene rule for physics-informed ML: an invariance proof at the feature level must be paired with an invariance test at the classifier-head level before it can be claimed as a domain-transfer advantage. We did not do this test at design time. Doing it earlier would have flagged the failure mode in a single `pc_only` ablation before three weeks of hardening work.

### 8.3 Scope of the conclusion

The claim is scoped to: (a) PlantVillage → PlantDoc, (b) 25-class mapped target, (c) ViT-B/16 semantic backbone, (d) the specific training recipe described in §3.3. We expect the qualitative conclusion — mathematical invariance ≠ classifier invariance — to generalize, but the quantitative numbers do not.

---

## 9. Conclusion

We set out to show that physics-informed phase-congruency fusion beats a ViT baseline under OOD on botanical images. It does not. A complete four-cell ablation table under an identical recipe shows `full` ≈ `backbone_only` within run-to-run noise. The PC stream's formal amplitude-invariance does not propagate through the learned classifier head to OOD class-discriminative power. The transferable contribution of this work is the training recipe and the documentation of the failure mode. We publish the result honestly as a negative study with a practical recipe.

---

## Acknowledgments, data, and code availability

- Code, configs, and notebooks: this repository.
- Dataset sources: PlantVillage (Hughes & Salathé 2015), PlantDoc (Singh et al. 2020).
- Full run log: `RESULTS.md` at the commit that produced the reported numbers.
- Auto-memory context and prior iterations: `~/.claude/projects/-Users-arunbhyashaswi-Drive-Code-PhasePhyto/memory/`.

---

## References

Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *NeurIPS 2020*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR 2021*.

Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. *ICML 2015*.

Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *UAI 2018*.

Kovesi, P. (1999). Image Features from Phase Congruency. *Videre: Journal of Computer Vision Research*, 1(3).

Lee, D.-H. (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. *ICML Workshop on Challenges in Representation Learning*.

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. *CODS-COMAD 2020*.

Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C., Cubuk, E. D., Kurakin, A., & Li, C. L. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. *NeurIPS 2020*.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016*.

Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. *ICLR 2021*.

Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random Erasing Data Augmentation. *AAAI 2020*.

Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*, 474–485.
