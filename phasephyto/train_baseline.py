"""Training entry point for a semantic-only timm baseline."""

import argparse
from ast import literal_eval
from contextlib import suppress
from pathlib import Path

import torch

from phasephyto.models import TimmClassifier
from phasephyto.train import build_dataloaders, build_loss
from phasephyto.training.trainer import Trainer
from phasephyto.utils.config import load_config
from phasephyto.utils.seed import seed_everything


def _parse_overrides(items: list[str] | None) -> dict:
    overrides: dict = {}
    if not items:
        return overrides

    for item in items:
        key, val = item.split("=", 1)
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        with suppress(ValueError, SyntaxError):
            val = literal_eval(val)
        d[parts[-1]] = val
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Train semantic-only timm baseline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", help="key=value overrides")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Override checkpoint directory (defaults to <cfg.checkpoint_dir>/baseline)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, _parse_overrides(args.override))
    seed_everything(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, num_classes = build_dataloaders(cfg)
    model = TimmClassifier(
        num_classes=num_classes,
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained_backbone,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    checkpoint_dir = args.checkpoint_dir or str(Path(cfg.checkpoint_dir) / "baseline")
    print(f"Baseline: {cfg.model.backbone_name} | Device: {device} | Classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=build_loss(cfg),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        grad_clip=cfg.training.grad_clip,
        patience=cfg.training.patience,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_wandb=cfg.use_wandb,
        project_name=f"{cfg.project_name}-baseline",
    )
    trainer.fit()
    print(f"\nBaseline training complete. Best val F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
