#!/usr/bin/env python3
"""Prepare strict apple-overlap datasets across PV, PlantDoc, and PP2021.

This script builds a shared 3-class label space using PlantVillage-compatible
class names:

- Apple___healthy
- Apple___Apple_scab
- Apple___Cedar_apple_rust

Inputs are expected to already exist locally (for example after using
`scripts/download_data.py`). PlantDoc and Plant Pathology 2021 labels are
normalized into the PlantVillage naming convention, and the output is written as
ImageFolder layout suitable for the existing PhasePhyto train/eval pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasephyto.data.class_mapping import (  # noqa: E402
    APPLE_STRICT_CLASSES,
    PLANTDOC_TO_PLANTVILLAGE,
    canonicalize_plant_pathology_2021_class,
)
from phasephyto.data.splits import (  # noqa: E402
    IMAGE_EXTENSIONS,
    resolve_image_folder,
)

APPLE_STRICT_SET = set(APPLE_STRICT_CLASSES)
PLANTDOC_APPLE_MAP = {
    raw: mapped for raw, mapped in PLANTDOC_TO_PLANTVILLAGE.items() if mapped in APPLE_STRICT_SET
}


def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _copy_class_images(
    src_dir: Path,
    dst_dir: Path,
    *,
    mode: str,
    prefix: str | None = None,
) -> int:
    copied = 0
    for image_path in sorted(src_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        filename = f"{prefix}__{image_path.name}" if prefix else image_path.name
        _materialize_file(image_path.resolve(), dst_dir / filename, mode)
        copied += 1
    return copied


def _prepare_plantvillage(source_root: Path, output_root: Path, *, mode: str) -> dict[str, int]:
    resolved = resolve_image_folder(source_root, ("train", "training", "val", "test"))
    out_root = output_root / "plantvillage"
    summary: dict[str, int] = {}
    for class_name in APPLE_STRICT_CLASSES:
        class_dir = resolved / class_name
        if not class_dir.exists():
            summary[class_name] = 0
            continue
        summary[class_name] = _copy_class_images(class_dir, out_root / class_name, mode=mode)
    return summary


def _prepare_plantdoc(source_root: Path, output_root: Path, *, mode: str) -> dict[str, int]:
    resolved = resolve_image_folder(source_root, ("test", "Test", "val", "valid"))
    out_root = output_root / "plantdoc"
    summary = {class_name: 0 for class_name in APPLE_STRICT_CLASSES}
    for raw_name, mapped_name in PLANTDOC_APPLE_MAP.items():
        class_dir = resolved / raw_name
        if not class_dir.exists():
            continue
        summary[mapped_name] += _copy_class_images(
            class_dir,
            out_root / mapped_name,
            mode=mode,
            prefix=raw_name.replace(" ", "_"),
        )
    return summary


def _prepare_plant_pathology_2021(
    source_root: Path,
    output_root: Path,
    *,
    mode: str,
) -> dict[str, int]:
    resolved = resolve_image_folder(source_root, ("train", "val", "valid", "test"))
    out_root = output_root / "plant_pathology_2021"
    summary = {class_name: 0 for class_name in APPLE_STRICT_CLASSES}
    for class_dir in sorted(p for p in resolved.iterdir() if p.is_dir()):
        mapped_name = canonicalize_plant_pathology_2021_class(class_dir.name)
        if mapped_name is None or mapped_name not in APPLE_STRICT_SET:
            continue
        summary[mapped_name] += _copy_class_images(
            class_dir,
            out_root / mapped_name,
            mode=mode,
            prefix=class_dir.name,
        )
    return summary


def prepare_apple_overlap(
    plantvillage_root: Path,
    plantdoc_root: Path,
    plant_pathology_2021_root: Path,
    output_root: Path,
    *,
    mode: str = "symlink",
    require_all_classes: bool = True,
    clean: bool = False,
) -> dict[str, Any]:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "labels": list(APPLE_STRICT_CLASSES),
        "datasets": {
            "plantvillage": _prepare_plantvillage(plantvillage_root, output_root, mode=mode),
            "plantdoc": _prepare_plantdoc(plantdoc_root, output_root, mode=mode),
            "plant_pathology_2021": _prepare_plant_pathology_2021(
                plant_pathology_2021_root,
                output_root,
                mode=mode,
            ),
        },
        "output_root": str(output_root),
        "mode": mode,
    }

    if require_all_classes:
        missing: list[str] = []
        for dataset_name, counts in report["datasets"].items():
            for class_name in APPLE_STRICT_CLASSES:
                if counts.get(class_name, 0) <= 0:
                    missing.append(f"{dataset_name}:{class_name}")
        if missing:
            raise RuntimeError(
                "Strict apple overlap is incomplete; missing images for: " + ", ".join(missing)
            )

    report_path = output_root / "overlap_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict apple-overlap datasets")
    parser.add_argument("--plantvillage", type=Path, required=True, help="PlantVillage root")
    parser.add_argument("--plantdoc", type=Path, required=True, help="PlantDoc root")
    parser.add_argument(
        "--plant-pathology-2021",
        type=Path,
        required=True,
        dest="plant_pathology_2021",
        help="Plant Pathology 2021 root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/overlap/apple_strict"),
        help="Output overlap root",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Materialization mode for overlap files",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow incomplete overlap subsets instead of failing",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing output root before rebuilding it",
    )
    args = parser.parse_args()

    report = prepare_apple_overlap(
        plantvillage_root=args.plantvillage,
        plantdoc_root=args.plantdoc,
        plant_pathology_2021_root=args.plant_pathology_2021,
        output_root=args.output,
        mode=args.mode,
        require_all_classes=not args.allow_missing,
        clean=args.clean,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
