#!/usr/bin/env python3
"""Audit class overlap between source and target image-folder datasets.

Use this before PlantVillage -> PlantDoc experiments to ensure the target
directory contains classes that the source-trained classifier can predict.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasephyto.data.splits import class_counts, resolve_image_folder  # noqa: E402


def normalize_class_name(name: str) -> str:
    """Normalize a class directory name for approximate dataset matching.

    Args:
        name: Raw class directory name.

    Returns:
        Lowercase alphanumeric token string with separators collapsed.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def audit_overlap(source: Path, target: Path) -> dict[str, Any]:
    """Build a class-overlap report for two image-folder roots.

    Args:
        source: Source-domain dataset root.
        target: Target-domain dataset root.

    Returns:
        JSON-serializable report containing raw and normalized overlap details.
    """
    source_root = resolve_image_folder(source, ("test", "val", "valid", "validation", "train"))
    target_root = resolve_image_folder(target, ("test", "val", "valid", "validation", "train"))
    source_counts = class_counts(source_root)
    target_counts = class_counts(target_root)

    source_norm = {normalize_class_name(name): name for name in source_counts}
    target_norm = {normalize_class_name(name): name for name in target_counts}
    common_norm = sorted(set(source_norm).intersection(target_norm))

    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_num_classes": len(source_counts),
        "target_num_classes": len(target_counts),
        "overlap_num_classes": len(common_norm),
        "overlap": [
            {
                "normalized": key,
                "source": source_norm[key],
                "target": target_norm[key],
                "source_images": source_counts[source_norm[key]],
                "target_images": target_counts[target_norm[key]],
            }
            for key in common_norm
        ],
        "source_only": sorted(source_norm[key] for key in set(source_norm) - set(target_norm)),
        "target_only": sorted(target_norm[key] for key in set(target_norm) - set(source_norm)),
    }


def main() -> None:
    """Run the class-overlap audit CLI."""
    parser = argparse.ArgumentParser(description="Audit source/target class overlap")
    parser.add_argument("--source", type=Path, required=True, help="Source dataset root")
    parser.add_argument("--target", type=Path, required=True, help="Target dataset root")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--fail-on-empty", action="store_true", help="Exit non-zero on no overlap")
    args = parser.parse_args()

    report = audit_overlap(args.source, args.target)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.fail_on_empty and report["overlap_num_classes"] == 0:
        raise SystemExit("No overlapping classes found.")


if __name__ == "__main__":
    main()
