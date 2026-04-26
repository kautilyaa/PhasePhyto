"""YAML configuration loading with dataclass validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backbone_name: str = "vit_base_patch16_224"
    fusion_dim: int = 256
    pc_scales: int = 4
    pc_orientations: int = 6
    num_heads: int = 4
    dropout: float = 0.1
    pretrained_backbone: bool = True
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 50
    warmup_epochs: int = 5
    batch_size: int = 32
    grad_clip: float = 1.0
    patience: int = 10
    loss: str = "cross_entropy"  # "cross_entropy", "focal", "label_smoothing"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0


@dataclass
class DataConfig:
    root: str = "data"
    use_case: str = "plant_disease"
    # Supported: plant_disease, cassava, plant_pathology_2021, rocole,
    # rice_leaf, banana_leaf, histology, pollen, wood.
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.2
    # Use-case-specific
    stain: str = "all"        # for histology
    domain: str = "all"       # for wood
    source_dir: str = ""      # source-domain root used for training and source eval
    target_dir: str = ""      # target-domain root used only for final OOD eval
    train_dir: str = ""       # optional explicit source train split
    val_dir: str = ""         # optional explicit source validation split
    eval_source_dir: str = "" # optional explicit source test/eval split
    eval_target_dir: str = "" # optional explicit target test/eval split


@dataclass
class PhasePhytoConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    project_name: str = "phasephyto"


def _update_dataclass(dc: Any, overrides: dict) -> None:
    """Recursively update a dataclass from a dict."""
    for k, v in overrides.items():
        if hasattr(dc, k):
            attr = getattr(dc, k)
            if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                _update_dataclass(attr, v)
            else:
                setattr(dc, k, v)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> PhasePhytoConfig:
    """Load config from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: Dict of ``dotted.key=value`` overrides.

    Returns:
        Fully resolved ``PhasePhytoConfig``.
    """
    cfg = PhasePhytoConfig()

    if config_path is not None:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        _update_dataclass(cfg, raw)

    if overrides:
        _update_dataclass(cfg, overrides)

    return cfg
