"""Baseline semantic-only classifier for domain-shift comparisons."""

from typing import Any

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None  # type: ignore[assignment]


class TimmClassifier(nn.Module):
    """Thin timm classifier wrapper with the same output contract as PhasePhyto.

    The training and evaluation loops expect models to return ``{"logits": ...}``
    and may pass an ``x_clahe`` keyword.  This baseline ignores the CLAHE stream so
    PhasePhyto-vs-backbone experiments can reuse the same loaders and metrics.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required: pip install timm")

        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not any(head_name in name for head_name in ("head", "classifier", "fc")):
                    param.requires_grad = False

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_clahe: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run semantic-only classification, ignoring ``x_clahe``."""
        return {"logits": self.model(x_rgb)}

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters."""
        return {
            "baseline": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
