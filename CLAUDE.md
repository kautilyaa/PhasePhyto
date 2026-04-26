# Claude Handoff Summary

## Objective completed

Built and hardened a **Colab-first apple-overlap pipeline** for PhasePhyto that:

- uses **Drive tar archives** as the persistent storage layer,
- reuses existing archives when present,
- hydrates data to **Colab SSD** for faster training/evaluation,
- prepares a strict shared-label overlap across:
  - PlantVillage
  - PlantDoc
  - Plant Pathology 2021

Shared labels:

- `Apple___healthy`
- `Apple___Apple_scab`
- `Apple___Cedar_apple_rust`

## Main artifacts added/updated

### Core overlap workflow
- `scripts/prepare_overlap_datasets.py`
- `configs/apple_overlap_plantdoc.yaml`
- `configs/apple_overlap_pp2021.yaml`

### Additional dataset support
- `phasephyto/data/datasets.py`
- `phasephyto/data/registry.py`
- `phasephyto/data/class_mapping.py`
- `phasephyto/train.py`
- `phasephyto/evaluate.py`
- `phasephyto/utils/config.py`
- `scripts/download_data.py`
- `phasephyto/batch_inference_config.py`

### Colab notebooks
- `notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb`
- `notebooks/PhasePhyto_Batch_Inference.ipynb`

### Docs
- `README.md`
- `notebooks/README.md`

## Important Colab behavior

The apple-overlap notebook was made more self-contained because the Colab clone
may lag behind local edits. It now includes inline helper logic for:

- overlap coverage inspection,
- overlap subset building,
- overlap config generation,
- archive-first Drive/SSD flow.

It also:

- checks whether raw dataset tar archives already exist on Drive,
- checks whether `apple_strict.tar` already exists,
- avoids rebuilding unless forced,
- provides a preflight overlap-coverage report,
- supports `allow_partial_overlap`.

## Key config defaults

Recommended notebook defaults:

- `force_redownload = False`
- `recreate_archives = False`
- `rebuild_overlap = False`
- `allow_partial_overlap = False`
- `hydrate_overlap_to_ssd = True`

## Known user-facing failure that was fixed

Observed issues included:

- relative-path failures in Colab,
- stale repo clone vs local notebook/script mismatch,
- opaque `CalledProcessError`,
- stale in-memory helper definitions in notebook kernels,
- overlap failures without clear missing-class diagnostics.

The notebook now has helper-version guards and better overlap diagnostics.

## Verification status

Final local checks passed:

- `python -m py_compile ...`
- `ruff check ...`
- `pytest ...`
- notebook code-cell syntax checks

At the end of the pass:

- **16 tests passed**
- `notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb` syntax OK
- `notebooks/PhasePhyto_Batch_Inference.ipynb` syntax OK

## Recommended next action for the user

In Colab:

1. restart the runtime,
2. rerun `notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb` from the top,
3. let the notebook:
   - reuse existing raw tar archives,
   - inspect overlap coverage,
   - build/archive `apple_strict.tar` if missing,
   - hydrate overlap data to SSD,
   - train/evaluate if toggled on.
