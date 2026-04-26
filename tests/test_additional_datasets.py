"""Tests for newly added plant benchmark dataset support."""

from pathlib import Path

from PIL import Image

from phasephyto.data import (
    BananaLeafDiseaseDataset,
    PlantPathology2021Dataset,
    RiceLeafDiseaseDataset,
    RoCoLeDataset,
)
from phasephyto.data.registry import DATASET_MAP
from scripts.download_data import (
    _plant_pathology_2021_label_from_row,
    prepare_manual_imagefolder_dataset,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(10, 120, 40)).save(path)


def test_additional_datasets_registered_in_dataset_map() -> None:
    assert DATASET_MAP["plant_pathology_2021"] is PlantPathology2021Dataset
    assert DATASET_MAP["rocole"] is RoCoLeDataset
    assert DATASET_MAP["rice_leaf"] is RiceLeafDiseaseDataset
    assert DATASET_MAP["banana_leaf"] is BananaLeafDiseaseDataset


def test_additional_imagefolder_datasets_load(tmp_path: Path) -> None:
    specs = [
        (PlantPathology2021Dataset, "healthy"),
        (RoCoLeDataset, "rust"),
        (RiceLeafDiseaseDataset, "brown_spot"),
        (BananaLeafDiseaseDataset, "healthy_leaf"),
    ]
    for dataset_cls, class_name in specs:
        root = tmp_path / dataset_cls.__name__
        _write_image(root / class_name / "0.jpg")
        _write_image(root / class_name / "1.jpg")
        ds = dataset_cls(root=root)
        assert ds.num_classes == 1
        assert ds.classes == [class_name]
        assert len(ds) == 2


def test_plant_pathology_2021_label_parser_supports_string_and_one_hot_rows() -> None:
    assert _plant_pathology_2021_label_from_row({"labels": "rust complex"}) == "rust_complex"
    assert _plant_pathology_2021_label_from_row({
        "complex": "0",
        "frog_eye_leaf_spot": "1",
        "healthy": "0",
        "powdery_mildew": "0",
        "rust": "1",
        "scab": "0",
    }) == "frog_eye_leaf_spot__rust"


def test_prepare_manual_imagefolder_dataset_prefers_raw_like_folder(tmp_path: Path) -> None:
    source = tmp_path / "banana_bundle"
    _write_image(source / "raw" / "healthy" / "a.jpg")
    _write_image(source / "raw" / "diseased" / "b.jpg")
    _write_image(source / "augmented" / "healthy" / "c.jpg")
    _write_image(source / "augmented" / "healthy" / "d.jpg")
    _write_image(source / "augmented" / "healthy" / "e.jpg")

    out_root = tmp_path / "prepared"
    prepare_manual_imagefolder_dataset("banana_leaf", out_root, source)

    dest = out_root / "banana_leaf"
    assert sorted(p.name for p in dest.iterdir() if p.is_dir()) == ["diseased", "healthy"]
    assert sum(1 for _ in dest.rglob("*.jpg")) == 2
