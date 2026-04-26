#!/usr/bin/env python3
"""
Download or prepare datasets for PhasePhyto.

Supports:
  - PlantVillage (via Kaggle API)
  - PlantDoc (via GitHub clone)
  - Cassava Leaf Disease (via Kaggle competitions API; requires one-time
    competition-rules acceptance)
  - Plant Pathology 2021 / FGVC8 (via Kaggle competitions API; normalized into
    ImageFolder layout using either the labels column or one-hot disease columns)
  - RoCoLe / Rice leaf / Banana leaf disease datasets from a user-provided
    source directory or archive already downloaded from the official host
  - Synthetic test data (no download needed)

Usage examples:
    python scripts/download_data.py --dataset plantvillage --output data/plant_disease
    python scripts/download_data.py --dataset plantdoc --output data/plant_disease
    python scripts/download_data.py --dataset cassava --output data/plant_disease
    python scripts/download_data.py --dataset plant_pathology_2021 \
        --output data/plant_benchmarks
    python scripts/download_data.py --dataset rocole \
        --output data/plant_benchmarks --source ~/Downloads/rocole.zip
    python scripts/download_data.py --dataset rice_leaf \
        --output data/plant_benchmarks --source ~/Downloads/rice_leaf
    python scripts/download_data.py --dataset banana_leaf \
        --output data/plant_benchmarks --source ~/Downloads/banana_leaf.zip
    python scripts/download_data.py --dataset synthetic --output data/synthetic --num-classes 10
    python scripts/download_data.py --dataset all --output data/plant_disease
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Canonical class names for the Kaggle Cassava Leaf Disease Classification
# competition. Follows the PlantVillage `<Crop>___<Disease>` convention so
# the same class-alias mapper and evaluation protocol can be reused.
CASSAVA_LABEL_TO_CLASS = {
    "0": "Cassava___Bacterial_Blight",
    "1": "Cassava___Brown_Streak_Disease",
    "2": "Cassava___Green_Mottle",
    "3": "Cassava___Mosaic_Disease",
    "4": "Cassava___healthy",
}

PLANT_PATHOLOGY_2021_DISEASE_COLUMNS = (
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
)

MANUAL_SOURCE_DATASETS = {"rocole", "rice_leaf", "banana_leaf"}


def _normalize_combo_label(label: str) -> str:
    parts = [
        re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_")
        for part in re.split(r"[,;+|]", label)
        if part.strip()
    ]
    return "__".join(parts) if parts else "unknown"


def _count_images(root: Path) -> int:
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _class_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not root.exists():
        return counts
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(
            1
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if n:
            counts[class_dir.name] = n
    return counts


def _find_imagefolder_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and _class_counts(path):
            candidates.append(path)
    return candidates


def _select_best_imagefolder_candidate(
    root: Path,
    *,
    prefer_keywords: Iterable[str] = (),
) -> Path | None:
    keywords = tuple(k.lower() for k in prefer_keywords)
    candidates = _find_imagefolder_candidates(root)
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, int, str]:
        counts = _class_counts(path)
        name_bonus = 1 if any(k in str(path).lower() for k in keywords) else 0
        return (name_bonus, len(counts), sum(counts.values()), str(path))

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _copy_imagefolder(src_root: Path, dest_root: Path) -> tuple[int, int]:
    dest_root.mkdir(parents=True, exist_ok=True)
    copied_classes = 0
    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if not _class_counts(src_root).get(class_dir.name, 0):
            continue
        shutil.copytree(class_dir, dest_root / class_dir.name, dirs_exist_ok=True)
        copied_classes += 1
    return copied_classes, _count_images(dest_root)


def _expanded_source_root(source: Path) -> tuple[Path, Path | None]:
    if source.is_dir():
        return source, None

    tmpdir = Path(tempfile.mkdtemp(prefix="phasephyto_dataset_"))
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmpdir)
    elif source.suffix.lower() in {
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
    } or source.name.endswith(".tar.gz"):
        with tarfile.open(source) as tf:
            tf.extractall(tmpdir)
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError(f"Unsupported source archive format: {source}")
    return tmpdir, tmpdir


def _ensure_kaggle() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])


def _plant_pathology_2021_label_from_row(row: dict[str, str]) -> str:
    if row.get("labels"):
        return _normalize_combo_label(row["labels"])
    if row.get("label"):
        return _normalize_combo_label(row["label"])

    active = []
    for key in PLANT_PATHOLOGY_2021_DISEASE_COLUMNS:
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "1.0", "true", "yes"}:
            active.append(key)
    if active:
        return "__".join(active)
    raise ValueError(f"Could not infer Plant Pathology 2021 label from row: {row}")


def download_plantvillage(output_dir: Path) -> None:
    dest = output_dir / "plantvillage"
    if dest.exists() and any(dest.iterdir()):
        print(f"PlantVillage already exists at {dest}, skipping download.")
        return

    _ensure_kaggle()
    print("Downloading PlantVillage from Kaggle...")
    tmp = output_dir / "_plantvillage_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "abdallahalidev/plantvillage-dataset",
        "-p", str(tmp),
    ], check=True)

    zip_file = tmp / "plantvillage-dataset.zip"
    if zip_file.exists():
        print("Extracting...")
        subprocess.run(["unzip", "-q", str(zip_file), "-d", str(tmp)], check=True)

    candidates = list(tmp.rglob("color"))
    if not candidates:
        candidates = [c for c in tmp.rglob("*") if c.is_dir() and any(c.glob("*/*.jpg"))]

    if not candidates:
        print(f"WARNING: Could not find image directory. Check {tmp} manually.")
        return

    src = candidates[0]
    dest.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(src.iterdir()):
        if class_dir.is_dir():
            shutil.copytree(class_dir, dest / class_dir.name, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    num_classes = sum(1 for d in dest.iterdir() if d.is_dir())
    print(
        f"PlantVillage: {_count_images(dest)} images across "
        f"{num_classes} classes"
    )


def download_plantdoc(output_dir: Path) -> None:
    dest = output_dir / "plantdoc"
    if dest.exists() and any(dest.iterdir()):
        print(f"PlantDoc already exists at {dest}, skipping download.")
        return

    print("Downloading PlantDoc from GitHub...")
    tmp = output_dir / "_plantdoc_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        str(tmp),
    ], check=True)

    for subdir_name in ["train", "test"]:
        src = tmp / subdir_name
        if src.exists():
            dest_sub = dest / subdir_name
            dest_sub.mkdir(parents=True, exist_ok=True)
            for class_dir in sorted(src.iterdir()):
                if class_dir.is_dir():
                    shutil.copytree(class_dir, dest_sub / class_dir.name, dirs_exist_ok=True)

    if not (dest / "train").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for item in tmp.iterdir():
            if item.is_dir() and item.name not in {".git", "_plantdoc_tmp"}:
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"PlantDoc extracted to {dest} ({_count_images(dest)} images)")


def download_cassava(output_dir: Path) -> None:
    dest = output_dir / "cassava"
    if dest.exists() and any(dest.iterdir()):
        print(f"Cassava already exists at {dest}, skipping download.")
        return

    _ensure_kaggle()
    print("Downloading Cassava Leaf Disease Classification from Kaggle...")
    print(
        "(If this 403s, accept the competition rules in-browser first: "
        "https://www.kaggle.com/c/cassava-leaf-disease-classification)"
    )

    tmp = output_dir / "_cassava_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "cassava-leaf-disease-classification",
        "-p", str(tmp),
    ], check=True)

    zip_files = list(tmp.glob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"No zip file downloaded to {tmp}")
    subprocess.run(["unzip", "-q", str(zip_files[0]), "-d", str(tmp / "extracted")], check=True)

    extracted = tmp / "extracted"
    train_csv = extracted / "train.csv"
    images_dir = extracted / "train_images"
    label_map_path = extracted / "label_num_to_disease_map.json"
    if not train_csv.exists() or not images_dir.exists():
        raise RuntimeError(
            f"Unexpected Cassava layout in {extracted}; "
            "expected train.csv and train_images/."
        )

    label_to_class = dict(CASSAVA_LABEL_TO_CLASS)
    if label_map_path.exists():
        try:
            raw_map = json.loads(label_map_path.read_text())
            label_to_class = {
                str(k): _cassava_disease_to_class_name(v)
                for k, v in raw_map.items()
            }
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"WARN: could not parse {label_map_path} ({exc}); "
                "using built-in Cassava label map."
            )

    dest.mkdir(parents=True, exist_ok=True)
    for class_name in label_to_class.values():
        (dest / class_name).mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing = 0
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id")
            label = row.get("label")
            if not image_id or label is None:
                continue
            class_name = label_to_class.get(str(label))
            if class_name is None:
                continue
            src_img = images_dir / image_id
            if not src_img.exists():
                n_missing += 1
                continue
            shutil.copy2(src_img, dest / class_name / image_id)
            n_copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    n_classes = sum(1 for d in dest.iterdir() if d.is_dir() and any(d.iterdir()))
    print(f"Cassava: {n_copied} images across {n_classes} classes (missing: {n_missing})")


def download_plant_pathology_2021(output_dir: Path) -> None:
    dest = output_dir / "plant_pathology_2021"
    if dest.exists() and any(dest.iterdir()):
        print(f"Plant Pathology 2021 already exists at {dest}, skipping download.")
        return

    _ensure_kaggle()
    print("Downloading Plant Pathology 2021 from Kaggle...")
    print(
        "(If this 403s, accept the competition rules in-browser first: "
        "https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data)"
    )
    tmp = output_dir / "_plant_pathology_2021_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "plant-pathology-2021-fgvc8",
        "-p", str(tmp),
    ], check=True)

    zip_files = list(tmp.glob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"No zip file downloaded to {tmp}")
    subprocess.run(["unzip", "-q", str(zip_files[0]), "-d", str(tmp / "extracted")], check=True)

    extracted = tmp / "extracted"
    train_csv = next(iter(extracted.rglob("train.csv")), None)
    if train_csv is None:
        raise RuntimeError(f"Could not find train.csv under {extracted}")

    image_dirs = [
        p
        for p in extracted.rglob("*")
        if p.is_dir()
        and any(
            c.is_file() and c.suffix.lower() in IMAGE_EXTENSIONS
            for c in p.iterdir()
        )
    ]
    if not image_dirs:
        raise RuntimeError(f"Could not find extracted image directory under {extracted}")
    images_dir = max(image_dirs, key=lambda p: sum(1 for _ in p.iterdir()))

    dest.mkdir(parents=True, exist_ok=True)
    label_counts: dict[str, int] = {}
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        image_key = "image" if "image" in (reader.fieldnames or []) else "image_id"
        for row in reader:
            image_id = row.get(image_key)
            if not image_id:
                continue
            class_name = _plant_pathology_2021_label_from_row(row)
            src_img = images_dir / image_id
            if not src_img.exists():
                for ext in IMAGE_EXTENSIONS:
                    candidate = images_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
            if not src_img.exists():
                continue
            class_dir = dest / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst_name = src_img.name if src_img.suffix else f"{image_id}.jpg"
            shutil.copy2(src_img, class_dir / dst_name)
            label_counts[class_name] = label_counts.get(class_name, 0) + 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(
        f"Plant Pathology 2021: {sum(label_counts.values())} images "
        f"across {len(label_counts)} classes"
    )


def prepare_manual_imagefolder_dataset(
    dataset_name: str,
    output_dir: Path,
    source: Path,
    *,
    prefer_keywords: Iterable[str] = ("raw", "original", "train"),
) -> None:
    dest = output_dir / dataset_name
    if dest.exists() and any(dest.iterdir()):
        print(f"{dataset_name} already exists at {dest}, skipping preparation.")
        return
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    expanded_root, cleanup_root = _expanded_source_root(source)
    try:
        candidate = _select_best_imagefolder_candidate(
            expanded_root,
            prefer_keywords=prefer_keywords,
        )
        if candidate is None:
            raise RuntimeError(
                f"Could not find an ImageFolder-style class root under {expanded_root}. "
                "Point --source at an extracted directory or archive that "
                "contains class folders with images."
            )
        copied_classes, n_images = _copy_imagefolder(candidate, dest)
        print(
            f"Prepared {dataset_name}: {n_images} images across "
            f"{copied_classes} classes from {candidate}"
        )
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)


def _cassava_disease_to_class_name(disease: str) -> str:
    text = disease.strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    if text.lower().startswith("cassava "):
        text = text[len("cassava "):].strip()
    if text.lower() == "healthy":
        return "Cassava___healthy"
    slug = "_".join(text.split())
    return f"Cassava___{slug}"


def create_synthetic(
    output_dir: Path,
    num_classes: int = 10,
    train_per_class: int = 50,
    test_per_class: int = 10,
) -> None:
    for split, n_per_class, brightness_range in [
        ("plantvillage", train_per_class, (0.8, 1.2)),
        ("plantdoc", test_per_class, (0.3, 2.5)),
    ]:
        split_dir = output_dir / split
        for c in range(num_classes):
            class_dir = split_dir / f"class_{c:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
                freq = (c + 1) * 3
                x = np.linspace(0, freq * np.pi, 256)
                y = np.linspace(0, freq * np.pi, 256)
                xx, yy = np.meshgrid(x, y)
                pattern = ((np.sin(xx + c) * np.cos(yy) + 1) * 40).astype(np.uint8)
                channel = np.clip(img[:, :, c % 3].astype(int) + pattern, 0, 255)
                img[:, :, c % 3] = channel.astype(np.uint8)
                brightness = np.random.uniform(*brightness_range)
                img = np.clip(img.astype(float) * brightness, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(class_dir / f"{split}_{c:02d}_{i:04d}.png")
        print(
            f"Synthetic [{split}]: {_count_images(split_dir)} images, "
            f"{num_classes} classes at {split_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or prepare PhasePhyto datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "plantvillage",
            "plantdoc",
            "cassava",
            "plant_pathology_2021",
            "rocole",
            "rice_leaf",
            "banana_leaf",
            "synthetic",
            "all",
        ],
        help="Dataset to download or prepare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/plant_disease",
        help="Output directory",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Local source dir/archive for manual-source datasets",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes for synthetic data",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Training samples per class for synthetic data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).expanduser() if args.source else None

    if args.dataset in ("plantvillage", "all"):
        download_plantvillage(output_dir)
    if args.dataset in ("plantdoc", "all"):
        download_plantdoc(output_dir)
    if args.dataset in ("cassava", "all"):
        download_cassava(output_dir)
    if args.dataset in ("plant_pathology_2021", "all"):
        download_plant_pathology_2021(output_dir)

    if args.dataset in MANUAL_SOURCE_DATASETS:
        if source is None:
            raise SystemExit(
                f"--dataset {args.dataset} requires --source pointing to "
                "a local extracted directory or archive downloaded from "
                "the official dataset host."
            )
        prepare_manual_imagefolder_dataset(args.dataset, output_dir, source)

    if args.dataset == "synthetic":
        create_synthetic(output_dir, args.num_classes, args.samples_per_class)

    print("\nDone.")


if __name__ == "__main__":
    main()
