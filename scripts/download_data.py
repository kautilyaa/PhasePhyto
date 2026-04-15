#!/usr/bin/env python3
"""
Download and prepare datasets for PhasePhyto.

Supports:
  - PlantVillage (via Kaggle API)
  - PlantDoc (via GitHub release)
  - Synthetic test data (no download needed)

Usage:
    python scripts/download_data.py --dataset plantvillage --output data/plant_disease
    python scripts/download_data.py --dataset plantdoc --output data/plant_disease
    python scripts/download_data.py --dataset synthetic --output data/synthetic --num-classes 10
    python scripts/download_data.py --dataset all --output data/plant_disease
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def download_plantvillage(output_dir: Path) -> None:
    """Download PlantVillage dataset via Kaggle API.

    Requires kaggle.json credentials at ~/.kaggle/kaggle.json.
    Dataset: abdallahalidev/plantvillage-dataset (~1.5GB)
    """
    dest = output_dir / "plantvillage"
    if dest.exists() and any(dest.iterdir()):
        print(f"PlantVillage already exists at {dest}, skipping download.")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

    print("Downloading PlantVillage from Kaggle...")
    tmp = output_dir / "_plantvillage_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "abdallahalidev/plantvillage-dataset",
        "-p", str(tmp),
    ], check=True)

    # Unzip
    zip_file = tmp / "plantvillage-dataset.zip"
    if zip_file.exists():
        print("Extracting...")
        subprocess.run(["unzip", "-q", str(zip_file), "-d", str(tmp)], check=True)

    # Find the image directory (varies by dataset version)
    # Typically: plantvillage dataset/color/ or plantvillage dataset/segmented/
    candidates = list(tmp.rglob("color"))
    if not candidates:
        candidates = list(tmp.rglob("*"))
        candidates = [c for c in candidates if c.is_dir() and any(c.glob("*/*.jpg"))]

    if candidates:
        src = candidates[0]
        dest.mkdir(parents=True, exist_ok=True)
        for class_dir in sorted(src.iterdir()):
            if class_dir.is_dir():
                shutil.copytree(class_dir, dest / class_dir.name, dirs_exist_ok=True)
        print(f"PlantVillage extracted to {dest}")
    else:
        print(f"WARNING: Could not find image directory. Check {tmp} manually.")
        return

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)
    n_images = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.JPG"))
    n_classes = sum(1 for d in dest.iterdir() if d.is_dir())
    print(f"PlantVillage: {n_images} images across {n_classes} classes")


def download_plantdoc(output_dir: Path) -> None:
    """Download PlantDoc dataset from GitHub.

    Dataset: pratikkayal/PlantDoc-Dataset
    """
    dest = output_dir / "plantdoc"
    if dest.exists() and any(dest.iterdir()):
        print(f"PlantDoc already exists at {dest}, skipping download.")
        return

    print("Downloading PlantDoc from GitHub...")
    tmp = output_dir / "_plantdoc_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # Clone the repo (sparse checkout for just the images)
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        str(tmp),
    ], check=True)

    # Find the train/test directories
    for subdir_name in ["train", "test"]:
        src = tmp / subdir_name
        if src.exists():
            dest_sub = dest / subdir_name
            dest_sub.mkdir(parents=True, exist_ok=True)
            for class_dir in sorted(src.iterdir()):
                if class_dir.is_dir():
                    shutil.copytree(class_dir, dest_sub / class_dir.name, dirs_exist_ok=True)

    # If no train/test split, copy everything flat
    if not (dest / "train").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for item in tmp.iterdir():
            if item.is_dir() and item.name not in {".git", "_plantdoc_tmp"}:
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    n_images = (
        sum(1 for _ in dest.rglob("*.jpg"))
        + sum(1 for _ in dest.rglob("*.JPG"))
        + sum(1 for _ in dest.rglob("*.png"))
    )
    print(f"PlantDoc extracted to {dest} ({n_images} images)")


def create_synthetic(
    output_dir: Path,
    num_classes: int = 10,
    train_per_class: int = 50,
    test_per_class: int = 10,
) -> None:
    """Create synthetic dataset for pipeline testing.

    Generates images with class-specific frequency patterns that are learnable
    but vary in brightness to simulate domain shift.
    """
    for split, n_per_class, brightness_range in [
        ("plantvillage", train_per_class, (0.8, 1.2)),    # "lab" conditions
        ("plantdoc", test_per_class, (0.3, 2.5)),          # "field" conditions (wider brightness)
    ]:
        split_dir = output_dir / split
        for c in range(num_classes):
            class_dir = split_dir / f"class_{c:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_per_class):
                img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

                # Class-specific structural pattern (learnable by PC)
                freq = (c + 1) * 3
                x = np.linspace(0, freq * np.pi, 256)
                y = np.linspace(0, freq * np.pi, 256)
                xx, yy = np.meshgrid(x, y)
                pattern = ((np.sin(xx + c) * np.cos(yy) + 1) * 40).astype(np.uint8)

                channel = np.clip(img[:, :, c % 3].astype(int) + pattern, 0, 255)
                img[:, :, c % 3] = channel.astype(np.uint8)

                # Apply brightness variation (domain shift simulation)
                brightness = np.random.uniform(*brightness_range)
                img = np.clip(img.astype(float) * brightness, 0, 255).astype(np.uint8)

                Image.fromarray(img).save(class_dir / f"{split}_{c:02d}_{i:04d}.png")

        n = sum(1 for _ in split_dir.rglob("*.png"))
        print(f"Synthetic [{split}]: {n} images, {num_classes} classes at {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download PhasePhyto datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["plantvillage", "plantdoc", "synthetic", "all"],
                        help="Dataset to download")
    parser.add_argument("--output", type=str, default="data/plant_disease",
                        help="Output directory")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes for synthetic data")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Training samples per class for synthetic data")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("plantvillage", "all"):
        download_plantvillage(output_dir)

    if args.dataset in ("plantdoc", "all"):
        download_plantdoc(output_dir)

    if args.dataset == "synthetic":
        create_synthetic(output_dir, args.num_classes, args.samples_per_class)

    print("\nDone.")


if __name__ == "__main__":
    main()
