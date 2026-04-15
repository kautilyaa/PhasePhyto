"""Tests for source/target data protocol and split resolution."""

from pathlib import Path

from PIL import Image

from phasephyto.data.datasets import TransformSubset
from phasephyto.data.splits import resolve_image_folder
from phasephyto.train import build_dataloaders
from phasephyto.utils.config import PhasePhytoConfig


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(128, 64, 32)).save(path)


def test_resolve_image_folder_prefers_test_split_for_nested_target(tmp_path: Path) -> None:
    """A PlantDoc-style root should resolve to its concrete test split."""
    _write_image(tmp_path / "plantdoc" / "train" / "class_a" / "train.png")
    _write_image(tmp_path / "plantdoc" / "test" / "class_a" / "test.png")

    resolved = resolve_image_folder(tmp_path / "plantdoc", ("test", "val", "train"))

    assert resolved == tmp_path / "plantdoc" / "test"


def test_build_dataloaders_uses_source_only_validation(tmp_path: Path) -> None:
    """Training should not require or inspect the configured target domain."""
    for class_name in ["class_a", "class_b"]:
        for idx in range(3):
            _write_image(tmp_path / "source" / class_name / f"{idx}.png")

    cfg = PhasePhytoConfig()
    cfg.data.source_dir = str(tmp_path / "source")
    cfg.data.target_dir = str(tmp_path / "missing_target")
    cfg.data.image_size = 16
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.val_split = 0.34
    cfg.training.batch_size = 2

    train_loader, val_loader, num_classes = build_dataloaders(cfg)

    assert num_classes == 2
    assert len(train_loader.dataset) > 0
    assert isinstance(val_loader.dataset, TransformSubset)
    assert str(tmp_path / "missing_target") not in repr(val_loader.dataset)
