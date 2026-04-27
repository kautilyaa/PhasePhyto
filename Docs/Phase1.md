# MSML640 Spring 2026 - Phase 1 Submission Package
**Course:** MSML640 Spring 2026  
**Due Date:** See ELMS  
**Project:** PhasePhyto (OOD Plant Disease Classification)

---

## 1) Report Submission

**Format:** PDF, max 2 pages  
**Filename:** `Group_25_phase1.pdf`

### Cover Information (Required)
- **Group Number:** `25`
- **Team Members + UID:**  
  - `Arunbh Yashaswi - 121304537`  
  - `Matheshwara Annamalai Senthilkumar - <UID>`  

---

## Revised Problem Statement (1 paragraph)

Our project addresses out-of-distribution (OOD) plant disease classification, where models trained on controlled lab-style images (PlantVillage) underperform on real-world field images (PlantDoc and Plant Pathology 2021). We focus on strict apple-class overlap to ensure fair label alignment and evaluate how architecture and training strategies affect true field generalization. The objective is to build an evidence-driven, reproducible pipeline that improves real-world disease recognition performance and clarifies which interventions are practically effective versus theoretically appealing but ineffective.

---

## Restated Proposal Goal

Build and evaluate an OOD-robust plant disease classification pipeline that transfers better from source-domain training data to field-domain targets, with reproducible experiments and deployable workflow components.

---

## Project Idea Adjustments (if any)

- Shifted from architecture novelty emphasis to **evidence-based OOD improvements**.
- Added a strict shared-label benchmark across:
  - PlantVillage
  - PlantDoc
  - Plant Pathology 2021 (PP2021)
- Added focused fix sweeps:
  - **Fix A:** post-hoc calibration variants
  - **Fix B:** rebalanced sampling variants

---

## Progress Since Proposal (1-2 sentences)

We completed the strict apple-overlap benchmark pipeline, trained and evaluated baseline and fix variants, and consolidated reproducible results and comparisons. The project is approximately **80% complete**, with remaining work focused on multi-seed validation, final optimization, and deployment polish.

---

## Architecture Changes / New Training Strategies (1-2 sentences)

We kept the ViT-based backbone and evaluated intervention strategies rather than introducing entirely new architectures at this phase. The key strategy improvement came from **softer class rebalancing** (`balanced_sampler_power=0.5`), which improved macro-F1 without destabilizing deployment behavior.

---

## Performance Metrics Comparison (include accuracy/graphs)

### Key Metrics Table (Apple Overlap)

| Variant | PP2021 Accuracy | PP2021 Macro-F1 | PlantDoc Accuracy | PlantDoc Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline overlap | 0.7136 | 0.6813 | 0.8621 | 0.8632 |
| Fix B v1 (rebalance=1.0) | 0.7416 | 0.6969 | 0.8966 | 0.8965 |
| **Fix B v2 (rebalance=0.5)** | **0.7393** | **0.7015** | **0.8966** | **0.8965** |

### Recommended Graphs to include in PDF
1. Bar chart: PP2021 Macro-F1 for Baseline vs Fix A vs Fix B variants  
2. Per-class F1 comparison (`healthy`, `apple_scab`, `cedar_apple_rust`) for baseline vs best fix

---

## Testing Results (Real-World Conditions)

Real-world testing was performed on **PP2021 (n=11,310)**, which better reflects field conditions than lab-curated datasets. Results show meaningful robustness gains but still indicate a non-trivial residual domain gap, primarily due to feature-shift behavior (e.g., class confusion patterns), not only prior mismatch.

---

## Integration Status of Team Components

- Data preparation and strict overlap generation: **Integrated**
- Training/evaluation pipeline: **Integrated**
- Metrics aggregation and result synthesis: **Integrated**
- Final demo/deployment packaging: **In progress**

> If solo project: replace with “All pipeline components are integrated end-to-end in a single reproducible workflow.”

---

## Model Improvements

- PP2021 macro-F1 improved from **0.6813 -> 0.7015** using softer rebalancing.
- PlantDoc macro-F1 improved from **0.8632 -> 0.8965** (noting smaller sample size caveat).
- Findings indicate that sampling-based recipe tuning is currently more effective than calibration-only fixes for this setup.

---

## Deployment Progress

- Colab-first reproducible workflow established with archive-first data handling.
- Evaluation artifacts and comparison summaries are generated in standardized formats (JSON/CSV/MD/plots).
- Deployment readiness is near-final, pending final-phase optimization and presentation packaging.

---

## 4) Challenges & Solutions

### Challenge 1: Prior-shift hypothesis uncertainty
- **Issue:** Unsure whether class prior mismatch was the main source of performance drop.
- **Fix:** Ran post-hoc calibration variants (uniform and oracle prior tests).
- **Outcome:** Macro-F1 decreased or remained weaker than rebalancing variants, indicating prior correction alone is insufficient.

### Challenge 2: Class imbalance affecting minority disease classes
- **Issue:** Oversampling settings could overcorrect specific classes.
- **Fix:** Compared full inverse-frequency rebalancing vs softened rebalancing (`power=0.5`).
- **Outcome:** Softer rebalancing gave best overall macro-F1 and reduced harmful side effects.

### Challenge 3: Reproducibility across environments
- **Issue:** Inconsistent behavior across local and Colab workflow states.
- **Fix:** Standardized archive-first data flow and overlap generation, with consistent artifact outputs.
- **Outcome:** More stable reruns and cleaner experiment tracking.

---

## 5) Next Steps for Final Phase

### Remaining Integration Tasks
- Run additional seeds for error bars and statistical confidence.
- Finalize integrated demo flow for end-to-end reproducibility.

### Final Optimization / Deployment
- Evaluate one transductive adaptation path (e.g., TENT or small target fine-tune).
- Final checkpoint selection, report polishing, and deployment/demo packaging.

---

## 2) Video Submission

**Format:** MP4  
**Filename:** `GroupNumber_phase1.mp4`  
**Duration:** 1-2 minutes  
**Goal:** Show expected output/end results and current progress state

### Video Structure (Suggested 90 sec)

#### 0:00-0:10 — Intro
- Group number, members, project title
- One-line problem statement

#### 0:10-0:30 — Objective
- Explain OOD gap (PlantVillage -> field datasets)
- State proposal goal and updated direction

#### 0:30-0:55 — System/Pipeline Demo
- Show overlap dataset prep workflow
- Show training/evaluation flow and generated outputs

#### 0:55-1:20 — Results
- Present key metrics table/graph
- Highlight improvement: **PP2021 macro-F1 0.6813 -> 0.7015**

#### 1:20-1:35 — Challenges & Fixes
- Briefly mention 2 key problems and specific fixes

#### 1:35-1:50 — Final Phase Plan
- Multi-seed validation, optimization, deployment finalization

#### 1:50-2:00 — Closing
- “Project is ~80% complete; final phase focuses on robustness validation and deployment-ready delivery.”

---

## Submission Checklist

- [ ] Add final Group Number and UID details
- [ ] Convert report to 2-page PDF
- [ ] Ensure at least 1 metrics table + 1 graph included
- [ ] Record and export 1-2 minute MP4
- [ ] Verify filenames:
  - `GroupNumber_phase1.pdf`
  - `GroupNumber_phase1.mp4`
- [ ] Submit both files on ELMS before deadline

---

## Quick Notes for Grading Alignment

- Keep statements evidence-based and specific.
- Mention both **what improved** and **what remains unresolved**.
- Use concrete metrics and tie each claim to an experiment result.
- Keep report concise, readable, and within 2 pages.