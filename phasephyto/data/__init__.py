"""Convenience exports for dataset, split, transform, and class-mapping helpers.

The package keeps heavyweight deps such as torch/torchvision behind lazy imports
so light-weight utilities (for example class-mapping audits or notebook preflight
checks) can be imported without requiring the full training stack.
"""

from __future__ import annotations

from importlib import import_module

from .class_mapping import (
    APPLE_STRICT_CLASSES,
    PLANT_PATHOLOGY_2021_TO_PLANTVILLAGE,
    PLANTDOC_TO_PLANTVILLAGE,
    canonicalize_plant_pathology_2021_class,
    create_mapped_plantdoc_folder,
    mapped_plantdoc_overlap,
    normalize_class_name,
)
from .registry import DATASET_MAP
from .splits import (
    class_counts,
    find_split_root,
    has_direct_class_images,
    resolve_image_folder,
)

__all__ = [
    "PlantDiseaseDataset",
    "CassavaDataset",
    "PlantPathology2021Dataset",
    "RoCoLeDataset",
    "RiceLeafDiseaseDataset",
    "BananaLeafDiseaseDataset",
    "APPLE_STRICT_CLASSES",
    "HistologyDataset",
    "PollenDataset",
    "WoodDataset",
    "TransformSubset",
    "PLANTDOC_TO_PLANTVILLAGE",
    "PLANT_PATHOLOGY_2021_TO_PLANTVILLAGE",
    "canonicalize_plant_pathology_2021_class",
    "class_counts",
    "create_mapped_plantdoc_folder",
    "find_split_root",
    "has_direct_class_images",
    "mapped_plantdoc_overlap",
    "normalize_class_name",
    "resolve_image_folder",
    "get_train_transforms",
    "get_val_transforms",
    "clahe_preprocess",
    "DATASET_MAP",
]

_LAZY_IMPORTS = {
    "PlantDiseaseDataset": (".datasets", "PlantDiseaseDataset"),
    "CassavaDataset": (".datasets", "CassavaDataset"),
    "PlantPathology2021Dataset": (".datasets", "PlantPathology2021Dataset"),
    "RoCoLeDataset": (".datasets", "RoCoLeDataset"),
    "RiceLeafDiseaseDataset": (".datasets", "RiceLeafDiseaseDataset"),
    "BananaLeafDiseaseDataset": (".datasets", "BananaLeafDiseaseDataset"),
    "HistologyDataset": (".datasets", "HistologyDataset"),
    "PollenDataset": (".datasets", "PollenDataset"),
    "WoodDataset": (".datasets", "WoodDataset"),
    "TransformSubset": (".datasets", "TransformSubset"),
    "get_train_transforms": (".transforms", "get_train_transforms"),
    "get_val_transforms": (".transforms", "get_val_transforms"),
    "clahe_preprocess": (".transforms", "clahe_preprocess"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
