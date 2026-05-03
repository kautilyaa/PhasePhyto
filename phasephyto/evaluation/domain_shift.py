"""Domain shift evaluation protocol for PhasePhyto.

Trains on source domain, evaluates on target domain without fine-tuning
to measure zero-shot cross-domain generalisation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


@torch.no_grad()
def evaluate_domain_shift(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: str = "cuda",
    class_names: list[str] | None = None,
) -> dict[str, dict]:
    """Evaluate model on both source (in-distribution) and target (OOD) data.

    Args:
        model: Trained PhasePhyto model.
        source_loader: In-distribution test DataLoader.
        target_loader: Out-of-distribution target DataLoader.
        device: Compute device.
        class_names: Class label names for reporting.

    Returns:
        Dict with ``'source'`` and ``'target'`` metric dicts, plus ``'delta'``
        showing the accuracy drop.
    """
    model.eval()
    model.to(device)

    results: dict[str, dict] = {}
    for split_name, loader in [("source", source_loader), ("target", target_loader)]:
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            if len(batch) == 3:
                rgb, clahe, labels = batch
                rgb, clahe = rgb.to(device), clahe.to(device)
            else:
                rgb, labels = batch
                rgb = rgb.to(device)
                clahe = None

            output = model(rgb, x_clahe=clahe)
            preds = output["logits"].argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

        results[split_name] = compute_metrics(all_labels, all_preds, class_names)

    target_accuracy = float(results["target"]["accuracy"])
    source_accuracy = float(results["source"]["accuracy"])
    target_f1 = float(results["target"]["f1_macro"])
    source_f1 = float(results["source"]["f1_macro"])
    delta = target_accuracy - source_accuracy
    results["delta"] = {
        "accuracy_drop": delta,
        "f1_drop": target_f1 - source_f1,
    }

    return results
