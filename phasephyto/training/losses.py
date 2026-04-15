"""
Loss functions for PhasePhyto training.

Provides Focal Loss (for class-imbalanced datasets like PlantDoc) and
Label Smoothing Cross-Entropy (for regularisation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reduces the relative loss for well-classified examples, focusing
    training on hard negatives.

    Reference: Lin et al. (2017). "Focal Loss for Dense Object Detection."

    Args:
        alpha: Weighting factor (scalar or per-class tensor).
        gamma: Focusing parameter.  gamma=0 recovers standard CE.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model outputs.
            targets: (B,) integer class labels.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing for regularisation.

    Args:
        smoothing: Label smoothing factor (0 = no smoothing).
        reduction: ``'mean'`` or ``'sum'``.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model outputs.
            targets: (B,) integer class labels.
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
