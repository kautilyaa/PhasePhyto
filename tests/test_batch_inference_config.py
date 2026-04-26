from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from phasephyto.batch_inference_config import (
    CANONICAL_ABLATIONS,
    inspect_dataset,
    normalize_dataset_runs,
)


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    plantvillage = tmp_path / "plantvillage"
    plantdoc_test = tmp_path / "plantdoc" / "test"
    cassava = tmp_path / "cassava"
    rice_leaf = tmp_path / "rice_leaf"

    _touch_image(plantvillage / "Apple___healthy" / "a.jpg")
    _touch_image(plantvillage / "Tomato___healthy" / "b.jpg")
    _touch_image(plantdoc_test / "Apple leaf" / "c.jpg")
    _touch_image(plantdoc_test / "Tomato leaf" / "d.jpg")
    _touch_image(cassava / "Cassava___healthy" / "e.jpg")
    _touch_image(rice_leaf / "Brown_Spot" / "f.jpg")

    manifest = {
        "plantvillage_dir": str(plantvillage),
        "plantdoc_dir": str(tmp_path / "plantdoc"),
        "plantdoc_test_dir": str(plantdoc_test),
        "cassava_dir": str(cassava),
        "rice_leaf_dir": str(rice_leaf),
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_inspect_dataset_resolves_nested_plantdoc_test_split(
    tiny_manifest: Path,
) -> None:
    manifest = json.loads(tiny_manifest.read_text())
    info = inspect_dataset("plantdoc", manifest["plantdoc_dir"])
    assert info.resolved_root.endswith("plantdoc/test")
    assert info.num_images == 2
    assert info.num_classes == 2


def test_normalize_dataset_runs_uses_manifest_and_requires_all_variants(
    tmp_path: Path,
    tiny_manifest: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {}
    for ablation in CANONICAL_ABLATIONS:
        ckpt = ckpt_dir / f"{ablation}.pt"
        _touch_file(ckpt)
        ckpts[ablation] = str(ckpt)

    runs = {
        "plantdoc_suite": {
            "dataset_kind": "plantdoc",
            "checkpoints": ckpts,
        }
    }

    source_json = tmp_path / "class_to_idx.json"
    source_json.write_text("{}")
    normalized = normalize_dataset_runs(
        runs,
        manifest_path=tiny_manifest,
        default_class_to_idx_source=str(source_json),
    )
    assert len(normalized) == 1
    run = normalized[0]
    assert run.run_name == "plantdoc_suite"
    assert run.dataset_kind == "plantdoc"
    assert run.dataset_root.endswith("plantdoc/test")
    assert run.inspection.resolved_root.endswith("plantdoc/test")
    assert run.class_to_idx_source == str(source_json)
    assert {ck.ablation for ck in run.checkpoints} == set(CANONICAL_ABLATIONS)


def test_normalize_dataset_runs_defaults_class_to_idx_source_to_first_checkpoint(
    tmp_path: Path,
    tiny_manifest: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {}
    for ablation in CANONICAL_ABLATIONS:
        ckpt = ckpt_dir / f"{ablation}.pt"
        _touch_file(ckpt)
        ckpts[ablation] = {"path": str(ckpt), "name": f"demo_{ablation}"}

    runs = {
        "cassava_suite": {
            "dataset_kind": "cassava",
            "checkpoints": ckpts,
        }
    }
    normalized = normalize_dataset_runs(runs, manifest_path=tiny_manifest)
    run = normalized[0]
    assert run.class_to_idx_source.endswith("full.pt")
    assert run.inspection.num_images == 1


def test_inspect_dataset_supports_new_manifest_backed_kinds(
    tmp_path: Path,
    tiny_manifest: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {}
    for ablation in CANONICAL_ABLATIONS:
        ckpt = ckpt_dir / f"{ablation}.pt"
        _touch_file(ckpt)
        ckpts[ablation] = str(ckpt)

    source_json = tmp_path / "class_to_idx.json"
    source_json.write_text("{}")
    runs = {
        "rice_suite": {
            "dataset_kind": "rice_leaf",
            "checkpoints": ckpts,
        }
    }
    normalized = normalize_dataset_runs(
        runs,
        manifest_path=tiny_manifest,
        default_class_to_idx_source=str(source_json),
    )
    assert normalized[0].dataset_kind == "rice_leaf"
    assert normalized[0].inspection.num_images == 1


def test_normalize_dataset_runs_rejects_missing_class_to_idx_source(
    tmp_path: Path,
    tiny_manifest: Path,
) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {}
    for ablation in CANONICAL_ABLATIONS:
        ckpt = ckpt_dir / f"{ablation}.pt"
        _touch_file(ckpt)
        ckpts[ablation] = str(ckpt)

    runs = {
        "plantdoc_suite": {
            "dataset_kind": "plantdoc",
            "class_to_idx_source": str(tmp_path / "missing.json"),
            "checkpoints": ckpts,
        }
    }
    with pytest.raises(FileNotFoundError, match="class_to_idx_source"):
        normalize_dataset_runs(runs, manifest_path=tiny_manifest)


def test_normalize_dataset_runs_rejects_missing_required_variant(
    tmp_path: Path,
    tiny_manifest: Path,
) -> None:
    only_full = tmp_path / "full.pt"
    _touch_file(only_full)
    runs = {
        "broken_suite": {
            "dataset_kind": "plantdoc",
            "checkpoints": {"full": str(only_full)},
        }
    }
    with pytest.raises(ValueError, match="missing required variants"):
        normalize_dataset_runs(runs, manifest_path=tiny_manifest, require_all_variants=True)
