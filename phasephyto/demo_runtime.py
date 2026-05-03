"""Runtime helpers for the Streamlit demo / Hugging Face Space."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from .data.transforms import CLAHETransform, IMAGENET_MEAN, IMAGENET_STD
from .demo_compat import DemoCheckpointSpec, build_demo_model

PLANT_DISEASE_25_CLASSES: tuple[str, ...] = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
)

MODEL_LAYOUT: dict[str, dict[str, str]] = {
    "full": {
        "label": "Full PhasePhyto",
        "artifact_dir": "Full",
        "checkpoint": "best_phasephyto.pt",
        "remote_checkpoint": "full/best_phasephyto.pt",
        "ablation": "full",
        "result_subdir": "drive-download-20260502T223147Z-3-001",
    },
    "full_ema": {
        "label": "Full PhasePhyto (EMA)",
        "artifact_dir": "Full",
        "checkpoint": "final_ema_phasephyto.pt",
        "remote_checkpoint": "full/final_ema_phasephyto.pt",
        "ablation": "full",
        "result_subdir": "drive-download-20260502T223147Z-3-001",
    },
    "backbone_only": {
        "label": "Backbone Only",
        "artifact_dir": "Backbone only",
        "checkpoint": "best_phasephyto.pt",
        "remote_checkpoint": "backbone_only/best_phasephyto.pt",
        "ablation": "backbone_only",
        "result_subdir": "drive-download-20260502T223311Z-3-001",
    },
    "no_fusion": {
        "label": "No Fusion",
        "artifact_dir": "No_fusion",
        "checkpoint": "best_phasephyto.pt",
        "remote_checkpoint": "no_fusion/best_phasephyto.pt",
        "ablation": "no_fusion",
        "result_subdir": "drive-download-20260502T223242Z-3-001",
    },
    "baseline": {
        "label": "Baseline ViT",
        "artifact_dir": ".",
        "checkpoint": "baseline_vit.pt",
        "remote_checkpoint": "baseline/baseline_vit.pt",
        "ablation": "baseline",
        "result_subdir": "",
    },
    "pc_only": {
        "label": "PC Only (metrics only)",
        "artifact_dir": "PC only",
        "checkpoint": "best_phasephyto.pt",
        "remote_checkpoint": None,
        "ablation": "pc_only",
        "result_subdir": "drive-download-20260502T223439Z-3-001",
    },
}


@dataclass(frozen=True)
class ModelArtifact:
    key: str
    label: str
    artifact_root: Path
    checkpoint_path: Path | None
    remote_checkpoint_path: str | None
    plots_dir: Path | None
    results_dir: Path | None
    ablation: str
    can_infer: bool
    checkpoint_missing_reason: str | None = None


@dataclass(frozen=True)
class DemoSample:
    key: str
    label: str
    label_index: int
    image_path: Path
    source_plot_path: Path
    known_good_models: tuple[str, ...]


@dataclass(frozen=True)
class PredictionResult:
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: np.ndarray
    display_image: np.ndarray
    pc_maps: dict[str, np.ndarray] | None
    attention_overlay: np.ndarray | None
    checkpoint_spec: DemoCheckpointSpec


DEFAULT_EXTERNAL_ROOT = Path("/external_artifacts")
DEFAULT_INTERNAL_ROOT = Path("hf_assets/finalresults")
DEFAULT_SAMPLES_MANIFEST = Path("hf_assets/demo_samples.json")
DEFAULT_HF_MODEL_REPO = "Mathesh0803/phasephyto-weights"


def resolve_artifact_root() -> Path:
    env_root = os.environ.get("PHASEPHYTO_ASSETS_ROOT")
    if env_root:
        return Path(env_root)
    if DEFAULT_INTERNAL_ROOT.exists():
        return DEFAULT_INTERNAL_ROOT
    return DEFAULT_EXTERNAL_ROOT


def resolve_hf_model_repo() -> str:
    return os.environ.get("PHASEPHYTO_HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO)


@lru_cache(maxsize=1)
def load_demo_samples(manifest_path: str | os.PathLike[str] | None = None) -> tuple[DemoSample, ...]:
    path = Path(manifest_path) if manifest_path else DEFAULT_SAMPLES_MANIFEST
    if not path.exists():
        return ()
    payload = json.loads(path.read_text())
    samples = []
    for row in payload.get("samples", []):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        source_plot_path = Path(row["source_plot_path"])
        if not source_plot_path.is_absolute():
            source_plot_path = path.parent / source_plot_path
        samples.append(
            DemoSample(
                key=str(row["key"]),
                label=str(row["label"]),
                label_index=int(row["label_index"]),
                image_path=image_path,
                source_plot_path=source_plot_path,
                known_good_models=tuple(row.get("known_good_models", [])),
            )
        )
    return tuple(samples)


def discover_model_artifacts(root: Path) -> dict[str, ModelArtifact]:
    artifacts: dict[str, ModelArtifact] = {}
    model_repo = resolve_hf_model_repo()
    for key, cfg in MODEL_LAYOUT.items():
        artifact_root = root / cfg["artifact_dir"] if cfg["artifact_dir"] != "." else root
        checkpoint_path = artifact_root / cfg["checkpoint"]
        run_dir = artifact_root / cfg["result_subdir"] if cfg["result_subdir"] else None
        plots_dir = run_dir / "plots" if run_dir and (run_dir / "plots").exists() else None
        results_dir = run_dir / "results" if run_dir and (run_dir / "results").exists() else None
        remote_checkpoint_path = cfg.get("remote_checkpoint")
        can_infer = checkpoint_path.exists() or bool(remote_checkpoint_path and model_repo)
        reason = None
        if not can_infer:
            reason = "Checkpoint not available in artifact root"
        artifacts[key] = ModelArtifact(
            key=key,
            label=cfg["label"],
            artifact_root=artifact_root,
            checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
            remote_checkpoint_path=remote_checkpoint_path,
            plots_dir=plots_dir,
            results_dir=results_dir,
            ablation=cfg["ablation"],
            can_infer=can_infer,
            checkpoint_missing_reason=reason,
        )
    return artifacts


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def read_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text()


def available_plot_paths(model: ModelArtifact) -> dict[str, Path]:
    if model.plots_dir is None:
        return {}
    return {
        p.name: p
        for p in sorted(model.plots_dir.glob("*.png"))
        if p.is_file()
    }


def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    image = image.convert("RGB")
    img_np = np.array(image)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    rgb_tensor = tf(image).unsqueeze(0)
    clahe_np = CLAHETransform()(cv2.resize(img_np, (image_size, image_size)))
    clahe_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(clahe_np).unsqueeze(0)
    display_img = cv2.resize(img_np, (image_size, image_size))
    return rgb_tensor, clahe_tensor, display_img


def build_attention_overlay(attn_weights: torch.Tensor, image_size: int = 224) -> np.ndarray:
    attn = attn_weights[0].detach().cpu()
    if attn.dim() == 3:
        attn = attn.mean(dim=0)
    attn_map = attn.mean(dim=0).view(14, 14).numpy()
    overlay = cv2.resize(attn_map, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
    return overlay


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def resolve_checkpoint_file(model: ModelArtifact) -> Path:
    if model.checkpoint_path is not None and model.checkpoint_path.exists():
        return model.checkpoint_path
    if model.remote_checkpoint_path:
        return Path(
            hf_hub_download(
                repo_id=resolve_hf_model_repo(),
                filename=model.remote_checkpoint_path,
                repo_type="model",
            )
        )
    raise FileNotFoundError(f"No checkpoint available for {model.label}")


def load_demo_model(model: ModelArtifact, *, device: str | None = None) -> tuple[torch.nn.Module, DemoCheckpointSpec]:
    checkpoint = load_checkpoint_payload(resolve_checkpoint_file(model))
    demo_model, spec = build_demo_model(
        checkpoint,
        num_classes=len(PLANT_DISEASE_25_CLASSES),
        fallback_ablation=model.ablation,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    demo_model.load_state_dict(state_dict, strict=True)
    chosen_device = device or choose_device()
    demo_model.to(chosen_device).eval()
    return demo_model, spec


def predict_image(
    model: torch.nn.Module,
    spec: DemoCheckpointSpec,
    image: Image.Image,
    *,
    device: str,
) -> PredictionResult:
    rgb, clahe, display = preprocess_pil_image(image, image_size=spec.image_size)
    rgb = rgb.to(device)
    clahe = clahe.to(device)
    with torch.no_grad():
        output = model(
            rgb,
            x_clahe=clahe,
            return_maps=(spec.model_type == "phasephyto"),
            return_attn=(spec.model_type == "phasephyto" and spec.ablation == "full"),
            ablation=spec.ablation if spec.model_type == "phasephyto" else None,
        )
    logits = output["logits"]
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = PLANT_DISEASE_25_CLASSES[pred_idx]
    pc_maps = None
    attention_overlay = None
    if spec.model_type == "phasephyto":
        pc_maps = {}
        for key in ("pc_magnitude", "phase_symmetry", "oriented_energy"):
            tensor = output.get(key)
            if isinstance(tensor, torch.Tensor):
                pc_maps[key] = tensor[0, 0].detach().cpu().numpy()
        attn_weights = output.get("attn_weights")
        if isinstance(attn_weights, torch.Tensor):
            attention_overlay = build_attention_overlay(attn_weights, image_size=spec.image_size)
    return PredictionResult(
        predicted_index=pred_idx,
        predicted_label=pred_label,
        confidence=float(probs[pred_idx]),
        probabilities=probs,
        display_image=display,
        pc_maps=pc_maps,
        attention_overlay=attention_overlay,
        checkpoint_spec=spec,
    )
