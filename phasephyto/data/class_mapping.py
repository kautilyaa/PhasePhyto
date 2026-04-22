"""Class-name mappings between PlantDoc and PlantVillage folder conventions."""

from __future__ import annotations

import re
from pathlib import Path

from .splits import IMAGE_EXTENSIONS

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Grape leaf": "Grape___healthy",
    "grape leaf": "Grape___healthy",
    "Grape leaf black rot": "Grape___Black_rot",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Soybean leaf": "Soybean___healthy",
    # These mappings are useful when the source PlantVillage subset includes
    # these classes. They are ignored automatically when source classes are absent.
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}


def normalize_class_name(name: str) -> str:
    """Normalize class names for coarse class-name comparisons.

    Args:
        name: Raw class directory name.

    Returns:
        Lowercase alphanumeric token string with separators collapsed.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def mapped_plantdoc_overlap(
    source_counts: dict[str, int],
    target_counts: dict[str, int],
) -> list[dict[str, object]]:
    """Report PlantDoc classes that can be mapped to source PlantVillage classes.

    Args:
        source_counts: Source class image counts keyed by PlantVillage class name.
        target_counts: Target class image counts keyed by PlantDoc class name.

    Returns:
        Mapping report rows for target classes whose mapped source class exists.
    """
    rows: list[dict[str, object]] = []
    for target_name in sorted(target_counts):
        source_name = PLANTDOC_TO_PLANTVILLAGE.get(target_name)
        if source_name is None or source_name not in source_counts:
            continue
        rows.append({
            "source": source_name,
            "target": target_name,
            "source_images": source_counts[source_name],
            "target_images": target_counts[target_name],
        })
    return rows


def create_mapped_plantdoc_folder(
    raw_target_root: str | Path,
    source_classes: set[str],
    mapped_target_root: str | Path,
) -> list[dict[str, object]]:
    """Create a PlantVillage-compatible PlantDoc target folder with symlinks.

    Args:
        raw_target_root: PlantDoc split root containing raw PlantDoc class folders.
        source_classes: PlantVillage class names available in the trained source model.
        mapped_target_root: Output folder whose class directories use PlantVillage names.

    Returns:
        Rows describing mapped classes and linked image counts.
    """
    raw_root = Path(raw_target_root)
    mapped_root = Path(mapped_target_root)
    mapped_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for target_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        source_name = PLANTDOC_TO_PLANTVILLAGE.get(target_dir.name)
        if source_name is None or source_name not in source_classes:
            continue

        destination = mapped_root / source_name
        destination.mkdir(parents=True, exist_ok=True)
        linked = 0
        for image_path in sorted(target_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            link_path = destination / image_path.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(image_path)
                except OSError:
                    # Fall back to a tiny text-free copy only when symlinks are unavailable.
                    link_path.write_bytes(image_path.read_bytes())
            linked += 1

        if linked:
            rows.append({
                "source": source_name,
                "target": target_dir.name,
                "mapped_images": linked,
                "mapped_dir": str(destination),
            })

    return rows
