"""Training entry point for PhasePhyto."""

import argparse
from ast import literal_eval
from contextlib import suppress
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from phasephyto.data.datasets import TransformSubset
from phasephyto.data.registry import DATASET_MAP
from phasephyto.data.splits import find_split_root, resolve_image_folder
from phasephyto.data.transforms import get_train_transforms, get_val_transforms
from phasephyto.models import PhasePhyto
from phasephyto.training.losses import FocalLoss, LabelSmoothingCE
from phasephyto.training.trainer import Trainer
from phasephyto.utils.config import load_config
from phasephyto.utils.seed import seed_everything


def _dataset_kwargs(cfg) -> dict:
    """Return use-case-specific dataset keyword arguments.

    Args:
        cfg: Loaded PhasePhyto configuration.

    Returns:
        Keyword arguments for the configured dataset class.
    """
    if cfg.data.use_case == "histology":
        return {"stain": cfg.data.stain}
    if cfg.data.use_case == "wood":
        return {"domain": cfg.data.domain}
    return {}


def _source_root(cfg) -> Path:
    """Return the source-domain root used for training.

    Args:
        cfg: Loaded PhasePhyto configuration.

    Returns:
        Path to the configured source root, falling back to ``data.root``.
    """
    return Path(cfg.data.source_dir or cfg.data.root)


def _extract_dataset_labels(ds) -> list[int]:
    """Get integer labels for every sample without running image transforms.

    Fast path uses ``.samples`` (PlantDiseaseDataset and friends), with a
    Subset-aware unwrap. Falls back to indexing ``ds[i][-1]`` only if no
    ``.samples`` attribute is reachable, since that path runs the full
    transform per sample.
    """
    if isinstance(ds, Subset):
        base = ds.dataset
        if hasattr(base, "samples"):
            return [int(base.samples[i][1]) for i in ds.indices]
    if hasattr(ds, "samples"):
        return [int(s[1]) for s in ds.samples]
    return [int(ds[i][-1]) for i in range(len(ds))]


def build_dataloaders(cfg):
    """Create source-domain train and validation DataLoaders from config.

    Validation is resolved from ``data.val_dir`` or a source-domain validation
    split.  If no source validation split exists, a deterministic random split
    is made from the source training root.  The target domain is never used for
    early stopping in this function.
    """
    train_tf = get_train_transforms(cfg.data.image_size)
    val_tf = get_val_transforms(cfg.data.image_size)

    DatasetClass = DATASET_MAP[cfg.data.use_case]
    kwargs = _dataset_kwargs(cfg)

    source_root = _source_root(cfg)
    train_dir = (
        resolve_image_folder(cfg.data.train_dir, ("train", "training"))
        if cfg.data.train_dir
        else resolve_image_folder(source_root, ("train", "training"))
    )
    val_dir = (
        resolve_image_folder(cfg.data.val_dir, ("val", "valid", "validation"))
        if cfg.data.val_dir
        else find_split_root(source_root, ("val", "valid", "validation"))
    )

    if val_dir is not None:
        train_ds = DatasetClass(root=train_dir, transform=train_tf, **kwargs)
        val_ds = DatasetClass(root=val_dir, transform=val_tf, **kwargs)
        num_classes = train_ds.num_classes
    else:
        full_ds = DatasetClass(root=train_dir, transform=train_tf, **kwargs)
        num_classes = full_ds.num_classes
        val_size = max(1, int(len(full_ds) * cfg.data.val_split))
        train_size = max(1, len(full_ds) - val_size)
        if train_size + val_size > len(full_ds):
            train_size = len(full_ds) - val_size
        generator = torch.Generator().manual_seed(cfg.seed)
        train_ds, raw_val_ds = random_split(
            full_ds, [train_size, val_size], generator=generator
        )
        val_ds = TransformSubset(raw_val_ds, val_tf)

    if cfg.data.balanced_sampler:
        labels = _extract_dataset_labels(train_ds)
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        per_class_weight = 1.0 / counts.clamp_min(1).double()
        sample_weights = per_class_weight[torch.tensor(labels)]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )
        print(
            f"Balanced sampler enabled. Class counts: {counts.tolist()} -> "
            "uniform expected sampling."
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader, num_classes


def build_loss(cfg):
    """Create loss function from config."""
    if cfg.training.loss == "focal":
        return FocalLoss(gamma=cfg.training.focal_gamma)
    elif cfg.training.loss == "label_smoothing":
        return LabelSmoothingCE(smoothing=cfg.training.label_smoothing)
    return torch.nn.CrossEntropyLoss()


def main():
    parser = argparse.ArgumentParser(description="Train PhasePhyto")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    # Parse overrides like "training.lr=1e-4" into nested dict
    overrides = {}
    if args.override:
        for item in args.override:
            key, val = item.split("=", 1)
            parts = key.split(".")
            d = overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            # Try to parse as number/bool/list without executing arbitrary code.
            with suppress(ValueError, SyntaxError):
                val = literal_eval(val)
            d[parts[-1]] = val

    cfg = load_config(args.config, overrides)
    seed_everything(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = build_dataloaders(cfg)
    print(f"Dataset: {cfg.data.use_case} | Classes: {num_classes}")
    print(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)}"
    )

    model = PhasePhyto(
        num_classes=num_classes,
        backbone_name=cfg.model.backbone_name,
        fusion_dim=cfg.model.fusion_dim,
        pc_scales=cfg.model.pc_scales,
        pc_orientations=cfg.model.pc_orientations,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        pretrained_backbone=cfg.model.pretrained_backbone,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    param_counts = model.count_parameters()
    print("Parameter counts:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    criterion = build_loss(cfg)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        grad_clip=cfg.training.grad_clip,
        patience=cfg.training.patience,
        checkpoint_dir=cfg.checkpoint_dir,
        device=device,
        use_wandb=cfg.use_wandb,
        project_name=cfg.project_name,
    )

    trainer.fit()
    print(f"\nTraining complete. Best val F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
