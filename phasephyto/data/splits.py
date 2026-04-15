"""Dataset split and class-audit helpers for image-folder protocols.

The project uses source-domain training data and target-domain test data for
domain-shift experiments.  These helpers keep train/validation/test resolution
explicit so training does not accidentally early-stop on the target domain.
"""

from collections.abc import Iterable
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def has_image_files(path: str | Path) -> bool:
    """Return whether ``path`` directly contains supported image files.

    Args:
        path: Directory to inspect.

    Returns:
        ``True`` if at least one direct child is a supported image file.
    """
    root = Path(path)
    return root.exists() and any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        for child in root.iterdir()
    )


def has_direct_class_images(path: str | Path) -> bool:
    """Return whether ``path`` looks like an image-folder class root.

    Args:
        path: Candidate dataset root with immediate class subdirectories.

    Returns:
        ``True`` when at least one immediate subdirectory contains image files.
    """
    root = Path(path)
    return root.exists() and any(
        child.is_dir() and has_image_files(child) for child in root.iterdir()
    )


def find_split_root(root: str | Path, split_names: Iterable[str]) -> Path | None:
    """Find the first existing image-folder split under ``root``.

    Args:
        root: Dataset root that may contain split folders such as ``train`` or ``test``.
        split_names: Candidate split folder names in priority order.

    Returns:
        The first split path that looks like an image-folder class root, or ``None``.
    """
    base = Path(root)
    for split_name in split_names:
        candidate = base / split_name
        if has_direct_class_images(candidate):
            return candidate
    return None


def resolve_image_folder(root: str | Path, preferred_splits: Iterable[str] = ()) -> Path:
    """Resolve a dataset root to a concrete image-folder class root.

    Args:
        root: Either a direct class-root or a parent containing split folders.
        preferred_splits: Split names to try when ``root`` is not directly an
            image-folder class root.

    Returns:
        A path suitable for the project dataset classes.
    """
    base = Path(root)
    if has_direct_class_images(base):
        return base
    split = find_split_root(base, preferred_splits)
    return split or base


def class_counts(root: str | Path) -> dict[str, int]:
    """Count images per class in a resolved image-folder root.

    Args:
        root: Direct image-folder class root.

    Returns:
        Mapping of class directory name to direct image count.
    """
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
