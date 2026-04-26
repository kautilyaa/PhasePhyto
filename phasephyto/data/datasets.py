"""
Dataset classes for all PhasePhyto use cases.

Each dataset returns ``(rgb_tensor, clahe_tensor, label)`` when used with
the ``DualTransform`` pipeline, or ``(image, label)`` with a standard
torchvision transform.

Datasets:
    1. PlantDiseaseDataset       -- PlantVillage (lab) / PlantDoc (field)
    2. CassavaDataset           -- Kaggle Cassava Leaf Disease (single-crop)
    3. PlantPathology2021Dataset -- Apple foliar disease benchmark (normalized)
    4. RoCoLeDataset            -- Robusta coffee leaf disease benchmark
    5. RiceLeafDiseaseDataset   -- Rice leaf disease benchmark
    6. BananaLeafDiseaseDataset -- Banana leaf disease benchmark
    7. HistologyDataset         -- Multi-stain potato tuber microscopy
    8. PollenDataset            -- Pollen grain microscopy
    9. WoodDataset              -- XyloTron macroscopic wood anatomy
"""

from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any, cast

from PIL import Image

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - lightweight utility imports
    class Dataset:  # type: ignore[override]
        """Tiny fallback so dataset metadata can be imported without torch installed."""

        pass


class TransformSubset(Dataset):
    """Wraps a ``torch.utils.data.Subset`` to override the parent's transform.

    Fixes the common bug where ``random_split`` creates subsets that inherit
    the parent dataset's transform. Without this, a validation subset would
    use training augmentations.

    Args:
        subset: A ``Subset`` from ``random_split``.
        transform: The transform to apply instead of the parent's.
    """

    def __init__(self, subset: Dataset, transform: Callable | None = None):
        self.subset = subset
        self.transform = transform
        self.dataset = subset.dataset  # type: ignore[attr-defined]
        if hasattr(self.dataset, "classes"):
            self.classes = self.dataset.classes
        if hasattr(self.dataset, "num_classes"):
            self.num_classes = self.dataset.num_classes
        if hasattr(self.dataset, "class_to_idx"):
            self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(cast(Sized, self.subset))

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        real_idx = self.subset.indices[idx]  # type: ignore[attr-defined]
        path, label = self.subset.dataset.samples[real_idx]  # type: ignore[attr-defined]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            result = self.transform(image)
            if isinstance(result, tuple):
                return (*result, label)
            return result, label
        return image, label


class PlantDiseaseDataset(Dataset):
    """Plant disease dataset supporting PlantVillage and PlantDoc.

    Expects directory layout::

        root/
          class_name_1/
            img001.jpg
            img002.jpg
          class_name_2/
            ...

    Args:
        root: Path to the dataset root directory.
        transform: Callable transform (e.g. ``DualTransform``).
        class_to_idx: Optional pre-defined class mapping.  If None,
            inferred from subdirectory names sorted alphabetically.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        class_to_idx: dict[str, int] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform

        # Discover classes from subdirectories
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        self.classes = list(self.class_to_idx.keys())
        self.num_classes = len(self.classes)

        # Build sample list: (path, label)
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            if class_dir.name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            result = self.transform(image)
            if isinstance(result, tuple):
                # DualTransform returns (rgb_tensor, clahe_tensor)
                return (*result, label)
            return result, label

        return image, label


class CassavaDataset(PlantDiseaseDataset):
    """Kaggle Cassava Leaf Disease Classification dataset.

    Reuses the ``PlantDiseaseDataset`` ImageFolder loader because
    ``scripts/download_data.py`` reorganizes the Kaggle flat
    ``train_images/`` + ``train.csv`` layout into
    ``<root>/Cassava___<Disease>/<image>.jpg`` at download time.

    Classes are disjoint from PlantVillage (Cassava is not represented in
    PlantVillage), so cross-dataset evaluation uses a separate class head.
    The five expected classes are exposed via ``EXPECTED_CLASSES`` for
    downstream protocol checks.

    Args:
        root: Path to the Cassava dataset root directory.
        transform: Callable transform (e.g. ``DualTransform``).
        class_to_idx: Optional pre-defined class mapping.
    """

    EXPECTED_CLASSES: tuple[str, ...] = (
        "Cassava___Bacterial_Blight",
        "Cassava___Brown_Streak_Disease",
        "Cassava___Green_Mottle",
        "Cassava___Mosaic_Disease",
        "Cassava___healthy",
    )


class PlantPathology2021Dataset(PlantDiseaseDataset):
    """Normalized Plant Pathology 2021 (FGVC8) apple foliar disease dataset.

    This loader expects the dataset to be pre-arranged into ImageFolder layout
    (one directory per normalized label or label-combination). The bundled
    download/preparation helper reorganizes the Kaggle competition files into
    that layout before training/evaluation.
    """


class RoCoLeDataset(PlantDiseaseDataset):
    """RoCoLe robusta coffee leaf disease dataset in ImageFolder layout."""


class RiceLeafDiseaseDataset(PlantDiseaseDataset):
    """Rice leaf disease dataset in ImageFolder layout."""


class BananaLeafDiseaseDataset(PlantDiseaseDataset):
    """Banana leaf disease dataset in ImageFolder layout."""


class HistologyDataset(Dataset):
    """Multi-stain potato tuber histology dataset.

    Supports filtering by stain type for cross-stain evaluation protocols.

    Expects layout::

        root/
          safranin/
            class_1/ ...
            class_2/ ...
          toluidine/
            class_1/ ...
          lugol/
            class_1/ ...

    Args:
        root: Dataset root.
        stain: Stain type filter (``'safranin'``, ``'toluidine'``, ``'lugol'``,
            or ``'all'``).
        transform: Callable transform.
    """

    def __init__(
        self,
        root: str | Path,
        stain: str = "all",
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.stain = stain

        stain_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if stain != "all":
            stain_dirs = [d for d in stain_dirs if stain.lower() in d.name.lower()]

        # Collect all class names across stains
        all_classes: set[str] = set()
        for sd in stain_dirs:
            for cd in sd.iterdir():
                if cd.is_dir():
                    all_classes.add(cd.name)
        self.classes = sorted(all_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.samples: list[tuple[Path, int]] = []
        for sd in stain_dirs:
            for cd in sorted(sd.iterdir()):
                if not cd.is_dir() or cd.name not in self.class_to_idx:
                    continue
                label = self.class_to_idx[cd.name]
                for img in sorted(cd.glob("*")):
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff"}:
                        self.samples.append((img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            result = self.transform(image)
            if isinstance(result, tuple):
                return (*result, label)
            return result, label
        return image, label


class PollenDataset(Dataset):
    """Pollen grain microscopy dataset.

    Standard image-folder layout with one subdirectory per species/type.

    Args:
        root: Dataset root (class subdirectories).
        transform: Callable transform.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform

        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.samples: list[tuple[Path, int]] = []
        for cd in class_dirs:
            label = self.class_to_idx[cd.name]
            for img in sorted(cd.glob("*")):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff"}:
                    self.samples.append((img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            result = self.transform(image)
            if isinstance(result, tuple):
                return (*result, label)
            return result, label
        return image, label


class WoodDataset(Dataset):
    """XyloTron macroscopic wood anatomy dataset.

    Supports lab/field domain split for OOD evaluation.

    Expects layout::

        root/
          lab/
            species_1/ ...
            species_2/ ...
          field/
            species_1/ ...

    Args:
        root: Dataset root.
        domain: ``'lab'``, ``'field'``, or ``'all'``.
        transform: Callable transform.
    """

    def __init__(
        self,
        root: str | Path,
        domain: str = "all",
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform

        domain_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if domain != "all":
            domain_dirs = [d for d in domain_dirs if domain.lower() in d.name.lower()]

        all_classes: set[str] = set()
        for dd in domain_dirs:
            for cd in dd.iterdir():
                if cd.is_dir():
                    all_classes.add(cd.name)
        self.classes = sorted(all_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.samples: list[tuple[Path, int]] = []
        for dd in domain_dirs:
            for cd in sorted(dd.iterdir()):
                if not cd.is_dir() or cd.name not in self.class_to_idx:
                    continue
                label = self.class_to_idx[cd.name]
                for img in sorted(cd.glob("*")):
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff"}:
                        self.samples.append((img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            result = self.transform(image)
            if isinstance(result, tuple):
                return (*result, label)
            return result, label
        return image, label
