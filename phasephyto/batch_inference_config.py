"""Configuration and validation helpers for batch-inference notebooks.

These helpers keep notebook logic focused on model execution while centralizing
multi-dataset configuration normalization, dataset-root resolution, and basic
preflight checks for required checkpoint variants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _has_image_files(path: str | Path) -> bool:
    root = Path(path)
    return root.exists() and any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        for child in root.iterdir()
    )


def _has_direct_class_images(path: str | Path) -> bool:
    root = Path(path)
    return root.exists() and any(
        child.is_dir() and _has_image_files(child) for child in root.iterdir()
    )


def _find_split_root(root: str | Path, split_names: Sequence[str]) -> Path | None:
    base = Path(root)
    for split_name in split_names:
        candidate = base / split_name
        if _has_direct_class_images(candidate):
            return candidate
    return None


def resolve_image_folder(root: str | Path, preferred_splits: Sequence[str] = ()) -> Path:
    base = Path(root)
    if _has_direct_class_images(base):
        return base
    split = _find_split_root(base, preferred_splits)
    return split or base


def class_counts(root: str | Path) -> dict[str, int]:
    base = Path(root)
    counts: dict[str, int] = {}
    if not base.exists():
        return counts
    for class_dir in sorted(child for child in base.iterdir() if child.is_dir()):
        count = sum(
            1
            for child in class_dir.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count:
            counts[class_dir.name] = count
    return counts


CANONICAL_ABLATIONS = ("full", "backbone_only", "no_fusion", "pc_only")
SUPPORTED_DATASET_KINDS = {
    "plantdoc",
    "cassava",
    "plantvillage",
    "plant_pathology_2021",
    "rocole",
    "rice_leaf",
    "banana_leaf",
    "custom",
}

_DATASET_MANIFEST_KEYS = {
    "plantdoc": ("plantdoc_test_dir", "plantdoc_dir"),
    "cassava": ("cassava_dir",),
    "plantvillage": ("plantvillage_dir",),
    "plant_pathology_2021": ("plant_pathology_2021_dir",),
    "rocole": ("rocole_dir",),
    "rice_leaf": ("rice_leaf_dir",),
    "banana_leaf": ("banana_leaf_dir",),
}

_PREFERRED_SPLITS = {
    "plantdoc": ("test", "Test", "val", "valid"),
    "cassava": ("test", "val", "valid", "train"),
    "plantvillage": ("test", "val", "valid", "validation", "train"),
    "plant_pathology_2021": ("val", "valid", "validation", "train"),
    "rocole": ("val", "valid", "validation", "train"),
    "rice_leaf": ("val", "valid", "validation", "train"),
    "banana_leaf": ("val", "valid", "validation", "train"),
    "custom": (),
}


@dataclass(frozen=True)
class NormalizedCheckpoint:
    name: str
    ablation: str
    path: str


@dataclass(frozen=True)
class DatasetInspection:
    dataset_kind: str
    requested_root: str
    resolved_root: str
    num_images: int
    num_classes: int
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedDatasetRun:
    run_name: str
    dataset_kind: str
    dataset_root: str
    class_to_idx_source: str
    checkpoints: tuple[NormalizedCheckpoint, ...]
    inspection: DatasetInspection


def _count_recursive_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_manifest(
    manifest_path: str | Path | None = None,
    drive_project_dir: str | Path | None = None,
) -> dict[str, Any]:
    candidate: Path | None = None
    if manifest_path:
        candidate = Path(manifest_path)
    elif drive_project_dir:
        candidate = (
            Path(drive_project_dir)
            / "data"
            / "plant_disease"
            / "dataset_manifest.json"
        )

    if candidate is None or not candidate.exists():
        return {}

    import json

    return json.loads(candidate.read_text())


def resolve_dataset_root(
    dataset_kind: str,
    dataset_root: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    drive_project_dir: str | Path | None = None,
) -> Path:
    """Resolve a concrete dataset root from explicit config or dataset manifest."""
    kind = dataset_kind.lower().strip()
    if kind not in SUPPORTED_DATASET_KINDS:
        raise ValueError(
            f"Unsupported dataset_kind={dataset_kind!r}; "
            f"expected one of {sorted(SUPPORTED_DATASET_KINDS)}"
        )

    if dataset_root:
        return Path(dataset_root)

    manifest = _load_manifest(manifest_path, drive_project_dir)
    for key in _DATASET_MANIFEST_KEYS.get(kind, ()):  # custom has no manifest fallback
        value = manifest.get(key)
        if value:
            return Path(value)

    raise FileNotFoundError(
        f"No dataset_root provided for {dataset_kind!r} and no manifest entry was found. "
        "Set dataset_root explicitly or pass manifest_path/drive_project_dir."
    )


def inspect_dataset(dataset_kind: str, dataset_root: str | Path) -> DatasetInspection:
    """Validate a dataset root and report its discovered image-folder stats."""
    kind = dataset_kind.lower().strip()
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if kind == "custom":
        resolved = root
        num_images = _count_recursive_images(root)
        if root.is_dir():
            class_dirs = [p for p in root.iterdir() if p.is_dir()]
            num_classes = len(class_dirs)
            class_names = tuple(sorted(p.name for p in class_dirs))
        else:
            num_classes = 0
            class_names = ()
    else:
        resolved = resolve_image_folder(root, _PREFERRED_SPLITS[kind])
        counts = class_counts(resolved)
        num_images = sum(counts.values())
        num_classes = len(counts)
        class_names = tuple(sorted(counts))

    if num_images == 0:
        raise ValueError(
            f"Dataset {dataset_kind!r} at {root} has no readable images "
            f"under resolved root {resolved}."
        )

    return DatasetInspection(
        dataset_kind=kind,
        requested_root=str(root),
        resolved_root=str(resolved),
        num_images=num_images,
        num_classes=num_classes,
        class_names=class_names,
    )


def normalize_checkpoints(
    checkpoints: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    require_all_variants: bool = True,
) -> tuple[NormalizedCheckpoint, ...]:
    """Normalize checkpoint config from dict or list form.

    Accepts either:
      - {"full": "/path/full.pt", ...}
      - {"full": {"path": "/path/full.pt", "name": "full_leafmask"}, ...}
      - [{"ablation": "full", "path": "/path/full.pt", "name": "..."}, ...]
    """
    entries: list[NormalizedCheckpoint] = []

    if isinstance(checkpoints, Mapping):
        iterable = checkpoints.items()
        for ablation, value in iterable:
            if isinstance(value, (str, Path)):
                name = str(ablation)
                path = str(value)
            elif isinstance(value, Mapping):
                name = str(value.get("name") or ablation)
                path = str(value.get("path") or "")
            else:
                raise TypeError(
                    f"Checkpoint entry for {ablation!r} must be a path "
                    f"or mapping, got {type(value).__name__}"
                )
            entries.append(
                NormalizedCheckpoint(
                    name=name,
                    ablation=str(ablation),
                    path=path,
                )
            )
    else:
        for idx, value in enumerate(checkpoints):
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"Checkpoint list entry #{idx} must be a mapping, "
                    f"got {type(value).__name__}"
                )
            ablation = str(value.get("ablation") or "")
            path = str(value.get("path") or "")
            name = str(value.get("name") or ablation)
            entries.append(NormalizedCheckpoint(name=name, ablation=ablation, path=path))

    if not entries:
        raise ValueError("No checkpoints configured.")

    seen_names: set[str] = set()
    seen_ablations: set[str] = set()
    normalized: list[NormalizedCheckpoint] = []
    for entry in entries:
        if entry.ablation not in CANONICAL_ABLATIONS:
            raise ValueError(
                f"Unsupported ablation {entry.ablation!r}; expected one of {CANONICAL_ABLATIONS}"
            )
        if not entry.path:
            raise ValueError(f"Checkpoint {entry.name!r} is missing a path.")
        if entry.name in seen_names:
            raise ValueError(f"Duplicate checkpoint name {entry.name!r}.")
        if entry.ablation in seen_ablations:
            raise ValueError(f"Duplicate checkpoint ablation {entry.ablation!r}.")
        if not Path(entry.path).exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {entry.path}")
        seen_names.add(entry.name)
        seen_ablations.add(entry.ablation)
        normalized.append(entry)

    if require_all_variants:
        missing = [ab for ab in CANONICAL_ABLATIONS if ab not in seen_ablations]
        if missing:
            raise ValueError(
                "Checkpoint set is missing required variants: " + ", ".join(missing)
            )

    return tuple(normalized)


def normalize_dataset_runs(
    eval_runs: Mapping[str, Mapping[str, Any]],
    *,
    manifest_path: str | Path | None = None,
    drive_project_dir: str | Path | None = None,
    default_class_to_idx_source: str | None = None,
    require_all_variants: bool = True,
) -> tuple[NormalizedDatasetRun, ...]:
    """Normalize and validate nested dataset->checkpoints notebook config."""
    if not isinstance(eval_runs, Mapping) or not eval_runs:
        raise ValueError("eval_runs must be a non-empty mapping of run_name -> spec")

    normalized_runs: list[NormalizedDatasetRun] = []
    for run_name, raw_spec in eval_runs.items():
        if not isinstance(raw_spec, Mapping):
            raise TypeError(
                f"Dataset run {run_name!r} must be a mapping, "
                f"got {type(raw_spec).__name__}"
            )

        dataset_kind = str(
            raw_spec.get("dataset_kind") or raw_spec.get("kind") or ""
        ).lower().strip()
        if dataset_kind not in SUPPORTED_DATASET_KINDS:
            raise ValueError(
                f"Dataset run {run_name!r} has unsupported dataset_kind={dataset_kind!r}; "
                f"expected one of {sorted(SUPPORTED_DATASET_KINDS)}"
            )

        dataset_root = resolve_dataset_root(
            dataset_kind,
            raw_spec.get("dataset_root"),
            manifest_path=raw_spec.get("manifest_path") or manifest_path,
            drive_project_dir=raw_spec.get("drive_project_dir") or drive_project_dir,
        )
        inspection = inspect_dataset(dataset_kind, dataset_root)

        checkpoints = normalize_checkpoints(
            raw_spec.get("checkpoints") or raw_spec.get("model_checkpoints") or (),
            require_all_variants=require_all_variants,
        )

        class_to_idx_source = str(
            raw_spec.get("class_to_idx_source")
            or default_class_to_idx_source
            or checkpoints[0].path
        )
        if not Path(class_to_idx_source).exists():
            raise FileNotFoundError(
                f"Dataset run {run_name!r} references "
                f"class_to_idx_source={class_to_idx_source!r}, "
                "but that path does not exist. Point it at a class_to_idx "
                "json, a source dataset root, "
                "or a checkpoint that embeds class_to_idx."
            )

        normalized_runs.append(
            NormalizedDatasetRun(
                run_name=str(run_name),
                dataset_kind=dataset_kind,
                dataset_root=str(dataset_root),
                class_to_idx_source=class_to_idx_source,
                checkpoints=checkpoints,
                inspection=inspection,
            )
        )

    return tuple(normalized_runs)
