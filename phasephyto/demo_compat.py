"""Notebook-compatible inference models for the PhasePhyto demo app.

These classes mirror the architecture used by the saved Colab checkpoints in
``Final Project/Finalresults``. The current repo model classes are close, but the
saved checkpoints include additional buffers/modules (``lum_weights``,
``aux_pc_head``) and ablation-specific forward behavior that are not represented
in the current lightweight training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None  # type: ignore[assignment]

from .models.cross_attention import StructuralSemanticFusion
from .models.illumination_norm import IlluminationNormStream
from .models.pc_encoder import PCEncoder
from .models.phase_congruency import LogGaborFilterBank, PhaseCongruencyExtractor
from .models.semantic_backbone import SemanticBackbone

Ablation = Literal["full", "pc_only", "backbone_only", "no_fusion"]
ABLATIONS: tuple[Ablation, ...] = ("full", "pc_only", "backbone_only", "no_fusion")


class NotebookCompatiblePhasePhyto(nn.Module):
    """Notebook-compatible PhasePhyto model used for demo inference."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "vit_base_patch16_224",
        fusion_dim: int = 256,
        pc_scales: int = 4,
        pc_orientations: int = 8,
        image_size: tuple[int, int] = (224, 224),
        num_heads: int = 8,
        dropout: float = 0.2,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        ablation: Ablation = "full",
    ):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}, got {ablation!r}")

        self.image_size = image_size
        self.ablation = ablation
        self.filter_bank = LogGaborFilterBank(image_size, pc_scales, pc_orientations)
        self.pc_extractor = PhaseCongruencyExtractor(pc_scales, pc_orientations)
        self.pc_encoder = PCEncoder(in_channels=3, fusion_dim=fusion_dim)
        self.backbone = SemanticBackbone(
            backbone_name=backbone_name,
            fusion_dim=fusion_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.illum_stream = IlluminationNormStream(out_dim=fusion_dim)
        self.fusion = StructuralSemanticFusion(
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_hidden_dim=fusion_dim * 2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.aux_pc_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.register_buffer(
            "lum_weights",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
        )

    def _resolve_ablation(self, ablation: Ablation | None) -> Ablation:
        if ablation is None:
            return self.ablation
        if ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}, got {ablation!r}")
        return ablation

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_clahe: torch.Tensor | None = None,
        return_maps: bool = False,
        return_attn: bool = False,
        return_aux: bool = False,
        ablation: Ablation | None = None,
    ) -> dict[str, torch.Tensor | str]:
        ablation = self._resolve_ablation(ablation)
        if x_clahe is None:
            x_clahe = x_rgb

        compute_pc = ablation in ("full", "pc_only", "no_fusion")
        compute_backbone = ablation in ("full", "backbone_only", "no_fusion")

        structural_tokens = None
        pc = None
        pc_input = None
        if compute_pc:
            gray = (x_rgb * self.lum_weights).sum(dim=1, keepdim=True)
            even, odd = self.filter_bank(gray)
            pc = self.pc_extractor(even, odd)
            pc_input = torch.cat(
                [pc["pc_magnitude"], pc["phase_symmetry"], pc["oriented_energy"]],
                dim=1,
            )
            structural_tokens = self.pc_encoder(pc_input)

        semantic_tokens = None
        if compute_backbone:
            semantic_tokens = self.backbone(x_rgb)

        illum = self.illum_stream(x_clahe)
        attn_weights = None

        if ablation == "full":
            if structural_tokens is None or semantic_tokens is None:
                raise RuntimeError("full ablation requires both PC and backbone streams")
            fused, attn_weights = self.fusion(
                structural_tokens,
                semantic_tokens,
                return_attention=return_attn,
            )
        elif ablation == "pc_only":
            if structural_tokens is None:
                raise RuntimeError("pc_only ablation requires structural tokens")
            fused = structural_tokens.mean(dim=1)
        elif ablation == "backbone_only":
            if semantic_tokens is None:
                raise RuntimeError("backbone_only ablation requires semantic tokens")
            fused = semantic_tokens.mean(dim=1)
        elif ablation == "no_fusion":
            if structural_tokens is None or semantic_tokens is None:
                raise RuntimeError("no_fusion ablation requires both streams")
            fused = 0.5 * (structural_tokens.mean(dim=1) + semantic_tokens.mean(dim=1))
        else:  # pragma: no cover
            raise ValueError(f"Unknown ablation: {ablation!r}")

        logits = self.classifier(torch.cat([fused, illum], dim=1))
        out: dict[str, torch.Tensor | str] = {"logits": logits, "ablation": ablation}
        if return_aux and structural_tokens is not None:
            out["aux_pc_logits"] = self.aux_pc_head(structural_tokens.mean(dim=1))
        if return_maps and pc is not None:
            out.update(pc)
            out["pc_input"] = pc_input
            out["fused"] = fused
        if return_attn and attn_weights is not None:
            out["attn_weights"] = attn_weights
        return out


class NotebookCompatibleBaseline(nn.Module):
    """Checkpoint-compatible baseline timm classifier.

    The saved baseline checkpoint uses the ``backbone.*`` prefix rather than the
    current repo's ``model.*`` wrapper.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required: pip install timm")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_clahe: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        del x_clahe
        return {"logits": self.backbone(x_rgb)}


@dataclass(frozen=True)
class DemoCheckpointSpec:
    model_type: Literal["phasephyto", "baseline"]
    ablation: str
    backbone_name: str
    fusion_dim: int | None = None
    pc_scales: int | None = None
    pc_orientations: int | None = None
    num_heads: int | None = None
    dropout: float | None = None
    image_size: int = 224


def infer_checkpoint_spec(checkpoint: dict[str, Any], *, fallback_ablation: str) -> DemoCheckpointSpec:
    """Infer a demo-loading spec from a saved checkpoint payload."""
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("backbone.") for key in state) and not any(
        key.startswith("filter_bank.") or key == "lum_weights" for key in state
    ):
        return DemoCheckpointSpec(
            model_type="baseline",
            ablation="baseline",
            backbone_name="vit_base_patch16_224",
        )

    cfg = checkpoint.get("config", {}) or {}
    return DemoCheckpointSpec(
        model_type="phasephyto",
        ablation=fallback_ablation,
        backbone_name=str(cfg.get("backbone_name", "vit_base_patch16_224")),
        fusion_dim=int(cfg.get("fusion_dim", 256)),
        pc_scales=int(cfg.get("pc_scales", 4)),
        pc_orientations=int(cfg.get("pc_orientations", 8)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.2)),
        image_size=int(cfg.get("image_size", 224)),
    )


def build_demo_model(
    checkpoint: dict[str, Any],
    *,
    num_classes: int,
    fallback_ablation: str,
) -> tuple[nn.Module, DemoCheckpointSpec]:
    """Instantiate a checkpoint-compatible inference model."""
    spec = infer_checkpoint_spec(checkpoint, fallback_ablation=fallback_ablation)
    if spec.model_type == "baseline":
        model = NotebookCompatibleBaseline(
            num_classes=num_classes,
            backbone_name=spec.backbone_name,
            pretrained=False,
        )
    else:
        model = NotebookCompatiblePhasePhyto(
            num_classes=num_classes,
            backbone_name=spec.backbone_name,
            fusion_dim=cast(int, spec.fusion_dim),
            pc_scales=cast(int, spec.pc_scales),
            pc_orientations=cast(int, spec.pc_orientations),
            image_size=(spec.image_size, spec.image_size),
            num_heads=cast(int, spec.num_heads),
            dropout=cast(float, spec.dropout),
            pretrained=False,
            ablation=cast(Ablation, spec.ablation),
        )
    return model, spec
