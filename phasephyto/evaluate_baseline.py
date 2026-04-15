"""Evaluate a semantic-only timm baseline on source and target domains."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phasephyto.evaluate import build_eval_datasets
from phasephyto.evaluation.domain_shift import evaluate_domain_shift
from phasephyto.models import TimmClassifier
from phasephyto.utils.config import load_config
from phasephyto.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic-only timm baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source-dir", type=str, help="Source domain test data")
    parser.add_argument("--target-dir", type=str, help="Target domain test data")
    parser.add_argument("--source-stain", type=str, help="Source stain for histology")
    parser.add_argument("--target-stain", type=str, help="Target stain for histology")
    parser.add_argument("--output", type=str, default="baseline_eval_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

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

    model = TimmClassifier(
        num_classes=source_ds.num_classes,
        backbone_name=cfg.model.backbone_name,
        pretrained=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_domain_shift(
        model,
        source_loader,
        target_loader,
        device=device,
        class_names=source_ds.classes,
    )
    results["use_case"] = cfg.data.use_case
    results["baseline"] = cfg.model.backbone_name

    print(f"BASELINE DOMAIN SHIFT EVALUATION: {cfg.model.backbone_name}")
    print(f"Source accuracy: {results['source']['accuracy']:.4f}")
    print(f"Target accuracy: {results['target']['accuracy']:.4f}")
    print(f"Accuracy delta: {results['delta']['accuracy_drop']:+.4f}")

    output = Path(args.output)
    with output.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
