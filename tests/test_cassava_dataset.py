"""Tests for the Cassava dataset loader."""

from pathlib import Path

from PIL import Image

from phasephyto.data import CassavaDataset
from phasephyto.train import DATASET_MAP


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(0, 128, 0)).save(path)


def test_cassava_dataset_loads_imagefolder_layout(tmp_path: Path) -> None:
    """CassavaDataset discovers class dirs and returns ``(image, label)``."""
    root = tmp_path / "cassava"
    for class_name in CassavaDataset.EXPECTED_CLASSES[:2]:
        for idx in range(2):
            _write_image(root / class_name / f"{idx}.jpg")

    ds = CassavaDataset(root=root)

    assert ds.num_classes == 2
    assert set(ds.classes) == set(CassavaDataset.EXPECTED_CLASSES[:2])
    assert len(ds) == 4

    image, label = ds[0]
    assert isinstance(image, Image.Image)
    assert label in (0, 1)


def test_cassava_dataset_registered_in_dataset_map() -> None:
    """The train-time dispatch table exposes ``cassava``."""
    assert DATASET_MAP["cassava"] is CassavaDataset


def test_cassava_expected_classes_cover_five_labels() -> None:
    """Kaggle Cassava has exactly 5 diagnostic classes."""
    assert len(CassavaDataset.EXPECTED_CLASSES) == 5
    assert all(name.startswith("Cassava___") for name in CassavaDataset.EXPECTED_CLASSES)
