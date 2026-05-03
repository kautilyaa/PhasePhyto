"""Evaluation entry point: load a checkpoint and evaluate on source + target domains.

Supports all 4 use cases via config-driven dataset dispatch.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phasephyto.data.datasets import PlantDiseaseDataset
from phasephyto.data.registry import DATASET_MAP
from phasephyto.data.splits import resolve_image_folder
from phasephyto.data.transforms import get_val_transforms
from phasephyto.evaluation.domain_shift import evaluate_domain_shift
from phasephyto.models import PhasePhyto
from phasephyto.train import _dataset_kwargs
from phasephyto.utils.config import load_config
from phasephyto.utils.seed import seed_everything


def build_eval_datasets(cfg, args):
    """Build source and target datasets from config + CLI overrides.

    Args:
        cfg: Loaded PhasePhyto configuration.
        args: Parsed evaluation arguments.

    Returns:
        Source and target datasets using the same class mapping.
    """
    val_tf = get_val_transforms(cfg.data.image_size)
    DatasetClass = DATASET_MAP[cfg.data.use_case]

    source_root = (
        args.source_dir
        or cfg.data.eval_source_dir
        or cfg.data.val_dir
        or cfg.data.source_dir
        or str(Path(cfg.data.root) / "val")
    )
    target_root = (
        args.target_dir
        or cfg.data.eval_target_dir
        or cfg.data.target_dir
        or str(Path(cfg.data.root) / "test")
    )
    source_dir = resolve_image_folder(
        source_root, ("test", "val", "valid", "validation")
    )
    target_dir = resolve_image_folder(
        target_root, ("test", "val", "valid", "validation")
    )

    # Use-case-specific kwargs
    source_kwargs: dict = _dataset_kwargs(cfg)
    target_kwargs: dict = _dataset_kwargs(cfg)

    if cfg.data.use_case == "histology":
        # Source: train stain, target: test stain
        source_kwargs["stain"] = (
            args.source_stain
            if hasattr(args, "source_stain") and args.source_stain
            else cfg.data.stain
        )
        target_kwargs["stain"] = (
            args.target_stain
            if hasattr(args, "target_stain") and args.target_stain
            else "all"
        )
    elif cfg.data.use_case == "wood":
        source_kwargs["domain"] = "lab"
        target_kwargs["domain"] = "field"

    source_ds = DatasetClass(root=source_dir, transform=val_tf, **source_kwargs)
    target_ds = DatasetClass(root=target_dir, transform=val_tf, **target_kwargs)

    # For plant_disease, align class mappings
    if cfg.data.use_case == "plant_disease" and hasattr(target_ds, "class_to_idx"):
        target_ds = PlantDiseaseDataset(
            root=target_dir, transform=val_tf, class_to_idx=source_ds.class_to_idx
        )
        if len(target_ds) == 0:
            raise ValueError(
                "Target dataset has no samples after applying the source class map. Run "
                "`python scripts/audit_class_overlap.py --source ... --target ...`."
            )

    return source_ds, target_ds


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhasePhyto")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source-dir", type=str, help="Source domain test data")
    parser.add_argument("--target-dir", type=str, help="Target domain test data")
    parser.add_argument("--source-stain", type=str, help="Source stain for histology")
    parser.add_argument("--target-stain", type=str, help="Target stain for histology")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    print(f"Use case: {cfg.data.use_case}")
    print(f"Device: {device}")

    source_ds, target_ds = build_eval_datasets(cfg, args)

    source_loader = DataLoader(
        source_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )

    print(f"Source samples: {len(source_ds)} | Target samples: {len(target_ds)}")
    print(f"Classes: {source_ds.num_classes}")

    model = PhasePhyto(
        num_classes=source_ds.num_classes,
        backbone_name=cfg.model.backbone_name,
        fusion_dim=cfg.model.fusion_dim,
        pc_scales=cfg.model.pc_scales,
        pc_orientations=cfg.model.pc_orientations,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

    results = evaluate_domain_shift(
        model,
        source_loader,
        target_loader,
        device=device,
        class_names=source_ds.classes,
    )

    print(f"\n{'='*60}")
    print(f"DOMAIN SHIFT EVALUATION: {cfg.data.use_case}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Source':<15} {'Target':<15} {'Delta':<15}")
    print(f"{'-'*65}")
    print(
        f"{'Accuracy':<20} {results['source']['accuracy']:<15.4f} "
        f"{results['target']['accuracy']:<15.4f} "
        f"{results['delta']['accuracy_drop']:<+15.4f}"
    )
    print(
        f"{'F1 (macro)':<20} {results['source']['f1_macro']:<15.4f} "
        f"{results['target']['f1_macro']:<15.4f} "
        f"{results['delta']['f1_drop']:<+15.4f}"
    )

    results["use_case"] = cfg.data.use_case
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
