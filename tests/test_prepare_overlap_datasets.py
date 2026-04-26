"""Tests for strict apple-overlap dataset preparation."""

from pathlib import Path

import pytest
from PIL import Image

from phasephyto.data.class_mapping import (
    APPLE_STRICT_CLASSES,
    canonicalize_plant_pathology_2021_class,
)
from scripts.prepare_overlap_datasets import (
    inspect_apple_overlap,
    prepare_apple_overlap,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(80, 140, 30)).save(path)


def test_canonicalize_plant_pathology_2021_class_handles_expected_labels() -> None:
    assert canonicalize_plant_pathology_2021_class("healthy") == "Apple___healthy"
    assert canonicalize_plant_pathology_2021_class("scab") == "Apple___Apple_scab"
    assert canonicalize_plant_pathology_2021_class("rust") == "Apple___Cedar_apple_rust"
    assert canonicalize_plant_pathology_2021_class("complex") is None


def test_inspect_apple_overlap_reports_missing_classes(tmp_path: Path) -> None:
    pv_root = tmp_path / "plantvillage"
    pd_root = tmp_path / "plantdoc" / "test"
    pp_root = tmp_path / "plant_pathology_2021"

    _write_image(pv_root / "Apple___healthy" / "pv_h.jpg")
    _write_image(pd_root / "Apple leaf" / "pd_h.jpg")
    _write_image(pp_root / "healthy" / "pp_h.jpg")

    report = inspect_apple_overlap(
        plantvillage_root=pv_root,
        plantdoc_root=tmp_path / "plantdoc",
        plant_pathology_2021_root=pp_root,
    )

    assert report["is_complete"] is False
    assert report["missing_by_dataset"]["plantvillage"] == [
        "Apple___Apple_scab",
        "Apple___Cedar_apple_rust",
    ]


def test_prepare_apple_overlap_builds_strict_three_class_subset(tmp_path: Path) -> None:
    pv_root = tmp_path / "plantvillage"
    pd_root = tmp_path / "plantdoc" / "test"
    pp_root = tmp_path / "plant_pathology_2021"

    _write_image(pv_root / "Apple___healthy" / "pv_h.jpg")
    _write_image(pv_root / "Apple___Apple_scab" / "pv_s.jpg")
    _write_image(pv_root / "Apple___Cedar_apple_rust" / "pv_r.jpg")
    _write_image(pv_root / "Tomato___healthy" / "ignore.jpg")

    _write_image(pd_root / "Apple leaf" / "pd_h.jpg")
    _write_image(pd_root / "Apple Scab Leaf" / "pd_s.jpg")
    _write_image(pd_root / "Apple rust leaf" / "pd_r.jpg")
    _write_image(pd_root / "Tomato leaf" / "ignore.jpg")

    _write_image(pp_root / "healthy" / "pp_h.jpg")
    _write_image(pp_root / "scab" / "pp_s.jpg")
    _write_image(pp_root / "rust" / "pp_r.jpg")
    _write_image(pp_root / "complex" / "ignore.jpg")

    out_root = tmp_path / "overlap" / "apple_strict"
    report = prepare_apple_overlap(
        plantvillage_root=pv_root,
        plantdoc_root=tmp_path / "plantdoc",
        plant_pathology_2021_root=pp_root,
        output_root=out_root,
        mode="copy",
        require_all_classes=True,
    )

    assert report["labels"] == list(APPLE_STRICT_CLASSES)
    assert (out_root / "overlap_manifest.json").exists()

    for dataset_name in ("plantvillage", "plantdoc", "plant_pathology_2021"):
        dataset_dir = out_root / dataset_name
        assert dataset_dir.exists()
        assert sorted(
            p.name for p in dataset_dir.iterdir() if p.is_dir()
        ) == sorted(APPLE_STRICT_CLASSES)
        for class_name in APPLE_STRICT_CLASSES:
            class_dir = dataset_dir / class_name
            assert any(class_dir.iterdir())


def test_prepare_apple_overlap_allow_missing_builds_partial_subset(tmp_path: Path) -> None:
    pv_root = tmp_path / "plantvillage"
    pd_root = tmp_path / "plantdoc" / "test"
    pp_root = tmp_path / "plant_pathology_2021"

    _write_image(pv_root / "Apple___healthy" / "pv_h.jpg")
    _write_image(pd_root / "Apple leaf" / "pd_h.jpg")
    _write_image(pp_root / "healthy" / "pp_h.jpg")

    out_root = tmp_path / "overlap_partial" / "apple_strict"
    report = prepare_apple_overlap(
        plantvillage_root=pv_root,
        plantdoc_root=tmp_path / "plantdoc",
        plant_pathology_2021_root=pp_root,
        output_root=out_root,
        mode="copy",
        require_all_classes=False,
    )

    assert report["is_complete"] is False
    assert (out_root / "plantvillage" / "Apple___healthy").exists()
    assert not (out_root / "plantvillage" / "Apple___Apple_scab").exists()


def test_prepare_apple_overlap_strict_mode_raises_clear_error(tmp_path: Path) -> None:
    pv_root = tmp_path / "plantvillage"
    pd_root = tmp_path / "plantdoc" / "test"
    pp_root = tmp_path / "plant_pathology_2021"

    _write_image(pv_root / "Apple___healthy" / "pv_h.jpg")
    _write_image(pd_root / "Apple leaf" / "pd_h.jpg")
    _write_image(pp_root / "healthy" / "pp_h.jpg")

    out_root = tmp_path / "overlap_fail" / "apple_strict"
    with pytest.raises(RuntimeError, match="Strict apple overlap is incomplete"):
        prepare_apple_overlap(
            plantvillage_root=pv_root,
            plantdoc_root=tmp_path / "plantdoc",
            plant_pathology_2021_root=pp_root,
            output_root=out_root,
            mode="copy",
            require_all_classes=True,
        )
