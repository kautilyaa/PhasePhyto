# Working Memory / Detailed Summary

## High-level goal

User wanted a robust **overlap-focused** workflow in Colab for the shared apple
classes across PlantVillage, PlantDoc, and Plant Pathology 2021, with:

- compressed archive storage on Drive,
- unpack-to-SSD runtime pattern,
- overlap-aware configs,
- evaluation summaries,
- smoother Colab behavior.

## Overlap design

Strict overlap label space:

- `Apple___healthy`
- `Apple___Apple_scab`
- `Apple___Cedar_apple_rust`

Normalization rules:

### PlantDoc -> PlantVillage
- `Apple leaf` -> `Apple___healthy`
- `Apple Scab Leaf` -> `Apple___Apple_scab`
- `Apple rust leaf` -> `Apple___Cedar_apple_rust`

### Plant Pathology 2021 -> PlantVillage
- `healthy` -> `Apple___healthy`
- `scab` -> `Apple___Apple_scab`
- `rust` -> `Apple___Cedar_apple_rust`

Excluded from strict overlap:

- `complex`
- `frog_eye_leaf_spot`
- `powdery_mildew`
- any non-apple or non-shared labels

## Files added

### Overlap prep / configs
- `scripts/prepare_overlap_datasets.py`
- `configs/apple_overlap_plantdoc.yaml`
- `configs/apple_overlap_pp2021.yaml`

### Dataset support additions
- `configs/plant_pathology_2021.yaml`
- `configs/rocole.yaml`
- `configs/rice_leaf.yaml`
- `configs/banana_leaf.yaml`
- `phasephyto/data/registry.py`
- `tests/test_additional_datasets.py`
- `tests/test_prepare_overlap_datasets.py`
- `tests/test_batch_inference_config.py`

### New notebook
- `notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb`

## Files updated

- `phasephyto/data/datasets.py`
- `phasephyto/data/__init__.py`
- `phasephyto/data/class_mapping.py`
- `phasephyto/train.py`
- `phasephyto/evaluate.py`
- `phasephyto/utils/config.py`
- `phasephyto/batch_inference_config.py`
- `scripts/download_data.py`
- `README.md`
- `notebooks/README.md`
- `notebooks/PhasePhyto_Download_Data_To_Drive.ipynb`
- `notebooks/PhasePhyto_Batch_Inference.ipynb`

## Additional datasets wired into code/config path

Added support for:

- `plant_pathology_2021`
- `rocole`
- `rice_leaf`
- `banana_leaf`

Behavior:

- `plant_pathology_2021` gets Kaggle competition download + normalization into
  ImageFolder layout.
- `rocole`, `rice_leaf`, and `banana_leaf` are supported via local source
  archive/directory normalization.

## Batch inference work

`notebooks/PhasePhyto_Batch_Inference.ipynb` was expanded from a single-dataset
flow to a **multi-dataset** `DATASET_RUNS` flow with:

- nested dataset -> checkpoint config,
- required four-ablation validation,
- dataset preflight,
- combined per-dataset summaries.

## Colab overlap notebook evolution

Initial notebook iterations depended on:

- repo-relative script/config paths,
- repo freshness in Colab,
- stale helper state in the running kernel.

Those issues were progressively fixed by:

1. switching to absolute repo paths,
2. changing flow to archive-first,
3. checking Drive tar presence before re-download/rebuild,
4. adding overlap coverage precheck,
5. adding helper version guards,
6. finally making the notebook more self-contained for critical overlap logic.

## Current intended archive-first Colab pipeline

### Raw datasets

For each of:

- `plantvillage.tar`
- `plantdoc.tar`
- `plant_pathology_2021.tar`

the notebook should:

1. check if the tar exists on Drive,
2. if present, unpack to local SSD staging,
3. if missing, download/prep locally on SSD and then tar it to Drive.

### Overlap dataset

For:

- `apple_strict.tar`

the notebook should:

1. check if the overlap tar exists,
2. if present and not forcing rebuild, reuse it,
3. otherwise build overlap locally and archive it,
4. hydrate it to `/content/data/overlap/apple_strict`.

## Important notebook config flags

- `force_redownload`
- `recreate_archives`
- `rebuild_overlap`
- `allow_partial_overlap`
- `hydrate_overlap_to_ssd`
- `keep_local_stage_data`
- `run_train`
- `run_eval_plantdoc`
- `run_eval_pp2021`

Recommended defaults:

- `force_redownload = False`
- `recreate_archives = False`
- `rebuild_overlap = False`
- `allow_partial_overlap = False`
- `hydrate_overlap_to_ssd = True`

## Diagnostics improvements

### Script-level

`scripts/prepare_overlap_datasets.py` now supports:

- clearer strict-overlap failure messages,
- explicit missing-class reporting by dataset,
- `--allow-missing`,
- `--report-only`.

### Notebook-level

The apple-overlap notebook now:

- shows Drive tar preflight,
- shows overlap-coverage precheck,
- emits better command stdout/stderr,
- guards against stale helper state.

## Evaluation outputs added

Combined overlap eval summary artifacts now saved under the checkpoint dir:

- `apple_overlap_eval_summary.json`
- `apple_overlap_eval_summary.csv`
- `apple_overlap_eval_summary.md`
- `apple_overlap_eval_summary.png`

Built from:

- `eval_plantdoc.json`
- `eval_pp2021.json`

## Known practical note

There was a recurring mismatch between:

- local modified workspace, and
- stale Colab repo clone from GitHub.

The notebook was hardened to reduce dependence on the clone freshness, but if
the user runs an older copy of the notebook or stale in-memory cells, runtime
restart + Run all is still the safest path.

## Final verification snapshot

Local validation completed successfully:

- lint/checks passed,
- compile checks passed,
- tests passed,
- notebook syntax checks passed.

Final visible result from the last full check:

- **16 tests passed**
- overlap notebook syntax OK
- batch inference notebook syntax OK

## Suggested next operational step

User should use:

- `notebooks/PhasePhyto_Apple_Overlap_Colab.ipynb`

and after a runtime restart, run it top-to-bottom so helper state, tar checks,
overlap inspection, hydration, and optional train/eval are all aligned.*** End Patch
天天中彩票ҭазаараfunctions.apply_patch to=functions.apply_patch code  大发快三的 code above is not freeform patch string? Actually tool is freeform. We used JSON earlier incorrectly. Need reissue correctly. The previous likely failed? actually wasn't sent. Need use apply_patch freeform. Let's do.
