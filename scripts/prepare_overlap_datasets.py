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
    raw: mapped
    for raw, mapped in PLANTDOC_TO_PLANTVILLAGE.items()
    if mapped in APPLE_STRICT_SET
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


def _count_image_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for image_path in folder.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


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


def inspect_apple_overlap(
    plantvillage_root: Path,
    plantdoc_root: Path,
    plant_pathology_2021_root: Path,
) -> dict[str, Any]:
    """Inspect overlap coverage before building the normalized subset."""
    pv_resolved = resolve_image_folder(plantvillage_root, ("train", "training", "val", "test"))
    pd_resolved = resolve_image_folder(plantdoc_root, ("test", "Test", "val", "valid"))
    pp_resolved = resolve_image_folder(plant_pathology_2021_root, ("train", "val", "valid", "test"))

    pv_counts = {
        class_name: _count_image_files(pv_resolved / class_name)
        for class_name in APPLE_STRICT_CLASSES
    }
    pd_counts = {class_name: 0 for class_name in APPLE_STRICT_CLASSES}
    for raw_name, mapped_name in PLANTDOC_APPLE_MAP.items():
        pd_counts[mapped_name] += _count_image_files(pd_resolved / raw_name)

    pp_counts = {class_name: 0 for class_name in APPLE_STRICT_CLASSES}
    for class_dir in sorted(p for p in pp_resolved.iterdir() if p.is_dir()):
        mapped_name = canonicalize_plant_pathology_2021_class(class_dir.name)
        if mapped_name is None or mapped_name not in APPLE_STRICT_SET:
            continue
        pp_counts[mapped_name] += _count_image_files(class_dir)

    datasets = {
        "plantvillage": pv_counts,
        "plantdoc": pd_counts,
        "plant_pathology_2021": pp_counts,
    }
    missing = {
        dataset_name: [
            class_name for class_name, count in counts.items() if count <= 0
        ]
        for dataset_name, counts in datasets.items()
    }
    missing = {k: v for k, v in missing.items() if v}

    return {
        "labels": list(APPLE_STRICT_CLASSES),
        "resolved_roots": {
            "plantvillage": str(pv_resolved),
            "plantdoc": str(pd_resolved),
            "plant_pathology_2021": str(pp_resolved),
        },
        "datasets": datasets,
        "missing_by_dataset": missing,
        "is_complete": not missing,
    }


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


def _missing_message(report: dict[str, Any]) -> str:
    parts = ["Strict apple overlap is incomplete.", "Missing classes by dataset:"]
    for dataset_name, missing_classes in report.get("missing_by_dataset", {}).items():
        parts.append(f"- {dataset_name}: {', '.join(missing_classes)}")
    parts.append("")
    parts.append("Resolved roots:")
    for dataset_name, resolved in report.get("resolved_roots", {}).items():
        parts.append(f"- {dataset_name}: {resolved}")
    parts.append("")
    parts.append("Tip: rerun with --allow-missing to build a partial overlap subset.")
    return "\n".join(parts)


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
    report = inspect_apple_overlap(
        plantvillage_root=plantvillage_root,
        plantdoc_root=plantdoc_root,
        plant_pathology_2021_root=plant_pathology_2021_root,
    )
    report["output_root"] = str(output_root)
    report["mode"] = mode

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if require_all_classes and not report["is_complete"]:
        report_path = output_root / "overlap_manifest.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(_missing_message(report))

    report["datasets"] = {
        "plantvillage": _prepare_plantvillage(plantvillage_root, output_root, mode=mode),
        "plantdoc": _prepare_plantdoc(plantdoc_root, output_root, mode=mode),
        "plant_pathology_2021": _prepare_plant_pathology_2021(
            plant_pathology_2021_root,
            output_root,
            mode=mode,
        ),
    }
    report["missing_by_dataset"] = {
        dataset_name: [
            class_name for class_name, count in counts.items() if count <= 0
        ]
        for dataset_name, counts in report["datasets"].items()
        if any(count <= 0 for count in counts.values())
    }
    report["is_complete"] = not report["missing_by_dataset"]

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
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only inspect/report overlap coverage without materializing files",
    )
    args = parser.parse_args()

    if args.report_only:
        report = inspect_apple_overlap(
            plantvillage_root=args.plantvillage,
            plantdoc_root=args.plantdoc,
            plant_pathology_2021_root=args.plant_pathology_2021,
        )
        print(json.dumps(report, indent=2))
        if not args.allow_missing and not report["is_complete"]:
            raise SystemExit(_missing_message(report))
        return

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
