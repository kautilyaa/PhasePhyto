# PhasePhyto Notebooks

Use the notebooks as separate workflow stages rather than one giant notebook.

## Quick order

Run the notebooks in this order:

1. **Optional but recommended first:** `PhasePhyto_Colab.ipynb` with synthetic data.
2. **One-time real-data setup:** `PhasePhyto_Download_Data_To_Drive.ipynb`.
3. **Real training/evaluation:** `PhasePhyto_Colab.ipynb` with Drive data paths.
4. **Inspect saved runs:** `PhasePhyto_Inspect_00_Index.ipynb`.
5. **Open focused inspectors as needed:** overview, metrics, plots, reports.

If you only want to verify the pipeline quickly, do step 1 only.
If you want real PlantVillage -> PlantDoc results, do all steps.

---

## Step 0: Colab runtime setup

For any notebook that trains or downloads data:

1. Open Google Colab.
2. Upload/open the notebook.
3. Select **Runtime -> Change runtime type -> T4 GPU** for training notebooks.
4. For the downloader notebook, GPU is not required, but it is fine if enabled.
5. Keep your Google account signed in so Drive mounting works.

---

## Step 1: Synthetic smoke test

Notebook:

`PhasePhyto_Colab.ipynb`

Use this before downloading real data.

Recommended config:

```python
CONFIG["use_synthetic"] = True
CONFIG["storage_backend"] = "drive"
CONFIG["drive_project_dir"] = "/content/drive/MyDrive/PhasePhyto"
CONFIG["epochs"] = 3  # optional quick test; use 20 for full synthetic run
```

Run all cells top-to-bottom.

Expected output in Drive:

```text
MyDrive/PhasePhyto/runs/<timestamp>/
  checkpoints/
  plots/
  results/
  run_manifest.json
```

Success criteria:

- notebook finishes without errors,
- `best_phasephyto.pt` exists,
- `phasephyto_results.json` exists,
- training curves are saved.

---

## Step 2: Download/prep real data once

Notebook:

`PhasePhyto_Download_Data_To_Drive.ipynb`

Purpose:

- Mounts Google Drive.
- Downloads PlantVillage via Kaggle.
- Downloads PlantDoc from GitHub.
- Stores reusable data under `MyDrive/PhasePhyto/data/plant_disease/`.
- Writes `dataset_manifest.json` with image counts and class-overlap audit.

Before running:

1. Get `kaggle.json` from Kaggle -> Account -> Create New API Token.
2. Either place it at:

```text
MyDrive/kaggle.json
```

or upload it when the notebook prompts you.

Default output:

```text
MyDrive/PhasePhyto/data/plant_disease/
  plantvillage/
  plantdoc/
    train/
    test/
  dataset_manifest.json
MyDrive/PhasePhyto/data/archives/
  plantvillage.tar
  plantdoc.tar
```

Success criteria:

- `plantvillage/` exists in Drive,
- `plantdoc/` exists in Drive,
- `dataset_manifest.json` exists,
- `plantvillage.tar` and `plantdoc.tar` exist in `MyDrive/PhasePhyto/data/archives/`,
- notebook prints paths to paste into the training notebook.

---

## Step 3: Real PlantVillage -> PlantDoc training/evaluation

Notebook:

`PhasePhyto_Colab.ipynb`

Use the Drive data from step 2.

Recommended config:

```python
CONFIG["use_synthetic"] = False
CONFIG["storage_backend"] = "drive"
CONFIG["drive_project_dir"] = "/content/drive/MyDrive/PhasePhyto"

CONFIG["hydrate_local_data_from_archives"] = True
CONFIG["drive_archive_dir"] = "/content/drive/MyDrive/PhasePhyto/data/archives"
CONFIG["local_data_root"] = "/content/data"

CONFIG["epochs"] = 20
CONFIG["batch_size"] = 16  # reduce to 8 if T4 runs out of memory
```

The notebook extracts:

```text
plantvillage.tar -> /content/data/plantvillage
plantdoc.tar     -> /content/data/plantdoc
```

and then automatically points training to those fast local SSD paths.

Run all cells top-to-bottom.

What this does:

- trains PhasePhyto on PlantVillage/source data,
- validates only on PlantVillage/source split,
- evaluates PlantDoc as target/OOD only,
- trains the ViT baseline on the same split,
- saves checkpoints, plots, metrics, reports, and manifest.

Expected output:

```text
MyDrive/PhasePhyto/runs/<timestamp>/
  checkpoints/
    best_phasephyto.pt
    baseline_vit.pt
  plots/
    training_curves.png
    illumination_invariance.png
    confusion_matrices.png
    analysis_sample_*.png
  results/
    phasephyto_domain_shift.json
    phasephyto_results.json
    target_classification_report.txt
  run_manifest.json
```

Success criteria:

- `run_manifest.json` exists,
- both checkpoints exist,
- `phasephyto_results.json` exists,
- `target_classification_report.txt` exists.

---

## Step 4: Inspect saved runs

Start with:

`PhasePhyto_Inspect_00_Index.ipynb`

Purpose:

- mounts Drive,
- lists available run folders,
- shows the latest run name.

If you want a specific run, copy its folder name and set this in any inspector:

```python
CONFIG["run_name"] = "<timestamp-folder-name>"
```

If `CONFIG["run_name"] = None`, inspectors load the latest run by folder name.

---

## Step 5: Open focused inspector notebooks

Use only the ones you need:

| Notebook | When to use |
|---|---|
| `PhasePhyto_Inspect_01_Run_Overview.ipynb` | Check manifest, artifact paths, and run folder health |
| `PhasePhyto_Inspect_02_Metrics.ipynb` | Compare PhasePhyto vs baseline metrics |
| `PhasePhyto_Inspect_03_Plots.ipynb` | View curves, confusion matrices, invariance plots, and sample analysis images |
| `PhasePhyto_Inspect_04_Reports.ipynb` | Read the target classification report |

`PhasePhyto_Inspect_Run_Results.ipynb` is retained only as a compatibility pointer to the split inspector workflow.

---

## Recommended real experiment workflow

For an actual result you might report:

1. Run synthetic smoke test with 3 epochs.
2. Run data downloader to Drive.
3. Run real training with 20 epochs.
4. Inspect metrics with `PhasePhyto_Inspect_02_Metrics.ipynb`.
5. Inspect plots with `PhasePhyto_Inspect_03_Plots.ipynb`.
6. Inspect report with `PhasePhyto_Inspect_04_Reports.ipynb`.
7. Only then copy real metrics into project docs.

Do **not** claim domain-shift improvement until the real run has produced saved
PhasePhyto-vs-baseline metrics in `phasephyto_results.json`.

---

## Tar / untar workflow for faster Colab setup

Google Drive is slow when Colab reads or copies thousands of small image files.
The faster workflow is to store one tar archive per dataset in Drive, then
extract those archives into Colab local SSD (`/content/data`) before training.

### Why tar helps

Slow pattern:

```text
Drive -> copy image1.jpg, image2.jpg, ... image50000.jpg -> /content/data
```

Faster pattern:

```text
Drive -> read plantvillage.tar once -> extract locally to /content/data
```

A tar file is just one large archive containing the dataset folder tree.

### Archive locations

The downloader notebook creates:

```text
MyDrive/PhasePhyto/data/archives/
  plantvillage.tar
  plantdoc.tar
```

The training notebook extracts them into:

```text
/content/data/
  plantvillage/
  plantdoc/
```

### Create tar archives manually

If you already have complete local folders in Colab:

```text
/content/data/plantvillage
/content/data/plantdoc
```

create Drive archives with:

```python
!mkdir -p /content/drive/MyDrive/PhasePhyto/data/archives

!tar -cf /content/drive/MyDrive/PhasePhyto/data/archives/plantvillage.tar \
  -C /content/data plantvillage

!tar -cf /content/drive/MyDrive/PhasePhyto/data/archives/plantdoc.tar \
  -C /content/data plantdoc
```

Explanation:

```bash
tar -cf output.tar -C /content/data plantvillage
```

means:

- `-c` = create archive,
- `-f output.tar` = write to this archive file,
- `-C /content/data` = change into `/content/data` first,
- `plantvillage` = archive the `plantvillage/` folder.

### Inspect tar contents

Before extracting, verify the archive has the expected folder root:

```python
!tar -tf /content/drive/MyDrive/PhasePhyto/data/archives/plantvillage.tar | head
!tar -tf /content/drive/MyDrive/PhasePhyto/data/archives/plantdoc.tar | head
```

Expected examples:

```text
plantvillage/
plantvillage/Apple___Apple_scab/...
```

```text
plantdoc/
plantdoc/train/...
plantdoc/test/...
```

### Extract tar archives manually

In a new Colab session:

```python
!mkdir -p /content/data

!tar -xf /content/drive/MyDrive/PhasePhyto/data/archives/plantvillage.tar \
  -C /content/data

!tar -xf /content/drive/MyDrive/PhasePhyto/data/archives/plantdoc.tar \
  -C /content/data
```

Explanation:

```bash
tar -xf archive.tar -C /content/data
```

means:

- `-x` = extract,
- `-f archive.tar` = read this archive file,
- `-C /content/data` = extract into `/content/data`.

### Recommended training config with automatic untar

The training notebook can do the extraction automatically. Use:

```python
CONFIG["use_synthetic"] = False
CONFIG["hydrate_local_data_from_archives"] = True
CONFIG["drive_archive_dir"] = "/content/drive/MyDrive/PhasePhyto/data/archives"
CONFIG["local_data_root"] = "/content/data"

CONFIG["storage_backend"] = "drive"
CONFIG["drive_project_dir"] = "/content/drive/MyDrive/PhasePhyto"
```

The notebook will extract:

```text
plantvillage.tar -> /content/data/plantvillage
plantdoc.tar     -> /content/data/plantdoc
```

and then train from local SSD paths.

### Verify extraction worked

After untarring:

```python
from pathlib import Path

PV = Path("/content/data/plantvillage")
PD = Path("/content/data/plantdoc")

print("PV exists:", PV.exists())
print("PD exists:", PD.exists())
print("PV files:", sum(1 for p in PV.rglob("*") if p.is_file()))
print("PD files:", sum(1 for p in PD.rglob("*") if p.is_file()))
print("PV class dirs:", len([p for p in PV.iterdir() if p.is_dir()]))
print("PD top dirs:", [p.name for p in PD.iterdir() if p.is_dir()][:10])
```

### Important rule

Do not train directly from the tar file. The tar file is only for fast storage
and transfer.

Correct flow:

```text
Drive tar archive
  -> extract to /content/data
  -> train from /content/data image folders
  -> save checkpoints/results back to Drive
```

---

## Handling corrupt or unreadable images

Large public image datasets sometimes contain corrupt files or files with image
extensions that PIL cannot decode. A typical error is:

```text
PIL.UnidentifiedImageError: cannot identify image file ... .JPG
```

If this happens, the model/training code is usually fine; one dataset file is
bad. Scan and remove/quarantine bad images before rerunning training.

### Scan local Colab data for bad images

Run this after extracting/copying data into `/content/data`:

```python
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


def find_bad_images(root):
    root = Path(root)
    bad = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]

    print(f"Checking {len(paths)} images under {root}...")
    for p in tqdm(paths):
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as exc:
            bad.append((p, repr(exc)))

    return bad


bad_pv = find_bad_images("/content/data/plantvillage")
bad_pd = find_bad_images("/content/data/plantdoc")

print("Bad PlantVillage images:", len(bad_pv))
print("Bad PlantDoc images:", len(bad_pd))

for p, err in bad_pv[:20]:
    print("PV BAD:", p, err)
for p, err in bad_pd[:20]:
    print("PD BAD:", p, err)
```

### Remove bad local files

For a quick fix, remove bad files only from the local Colab SSD copy:

```python
for p, err in bad_pv + bad_pd:
    print("Removing local bad image:", p)
    Path(p).unlink(missing_ok=True)
```

Then rerun the dataset/dataloader cell and restart training.

### Quarantine bad files in Drive too

If bad files came from the Drive master dataset, quarantine them so future tar
archives do not include them again:

```python
from pathlib import Path
import shutil

quarantine = Path("/content/drive/MyDrive/PhasePhyto/data/bad_images")
quarantine.mkdir(parents=True, exist_ok=True)

for local_path, err in bad_pv:
    local_path = Path(local_path)
    rel = local_path.relative_to("/content/data/plantvillage")
    drive_path = Path("/content/drive/MyDrive/PhasePhyto/data/plant_disease/plantvillage") / rel
    if drive_path.exists():
        dest = quarantine / "plantvillage" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(drive_path), str(dest))
        print("Quarantined Drive PV:", drive_path, "->", dest)

for local_path, err in bad_pd:
    local_path = Path(local_path)
    rel = local_path.relative_to("/content/data/plantdoc")
    drive_path = Path("/content/drive/MyDrive/PhasePhyto/data/plant_disease/plantdoc") / rel
    if drive_path.exists():
        dest = quarantine / "plantdoc" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(drive_path), str(dest))
        print("Quarantined Drive PD:", drive_path, "->", dest)
```

### Recreate tar archives after cleanup

After removing/quarantining bad files, recreate archives:

```python
!mkdir -p /content/drive/MyDrive/PhasePhyto/data/archives

!rm -f /content/drive/MyDrive/PhasePhyto/data/archives/plantvillage.tar
!rm -f /content/drive/MyDrive/PhasePhyto/data/archives/plantdoc.tar

!tar -cf /content/drive/MyDrive/PhasePhyto/data/archives/plantvillage.tar \
  -C /content/data plantvillage

!tar -cf /content/drive/MyDrive/PhasePhyto/data/archives/plantdoc.tar \
  -C /content/data plantdoc
```

### Quick one-file removal

If only one file fails and you want to continue quickly:

```python
from pathlib import Path

bad_file = Path("/content/data/plantvillage/Orange___Haunglongbing_(Citrus_greening)/5fa5d466-a1e2-4e65-9383-1f21fe54edde___UF.Citrus_HLB_Lab 0966.JPG")
print("Exists:", bad_file.exists())
print("Size:", bad_file.stat().st_size if bad_file.exists() else None)
bad_file.unlink(missing_ok=True)
print("Removed bad local file")
```

Scanning all images is safer because there may be more than one corrupt file.

### Speed note

If training from local `/content/data`, an epoch around 10--15 minutes on a T4
for ViT-B/16 is much more reasonable than multi-hour epochs from Drive. Keep
training reads local and save outputs to Drive.
