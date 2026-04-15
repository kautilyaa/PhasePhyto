"""
PhasePhyto: Full model assembly.

Three-stream architecture:
    1. Phase Congruency Stream   -> Structural Tokens (Q)
    2. Semantic Backbone Stream  -> Semantic Tokens (K, V)
    3. Illumination-Norm Stream  -> Auxiliary feature vector

Streams 1+2 are fused via cross-attention.  The fused vector is concatenated
with stream 3's auxiliary vector before the classification head.
"""

import torch
import torch.nn as nn

from .cross_attention import StructuralSemanticFusion
from .illumination_norm import IlluminationNormStream
from .pc_encoder import PCEncoder
from .phase_congruency import LogGaborFilterBank, PhaseCongruencyExtractor
from .semantic_backbone import SemanticBackbone


class PhasePhyto(nn.Module):
    """Complete PhasePhyto model.

    Args:
        num_classes: Number of output classification categories.
        backbone_name: timm model name for the semantic backbone.
        fusion_dim: Shared embedding dimension for cross-attention.
        pc_scales: Number of Log-Gabor wavelet scales.
        pc_orientations: Number of Log-Gabor filter orientations.
        image_size: Expected input spatial size (H, W).
        num_heads: Cross-attention heads.
        dropout: Dropout rate for fusion and classifier.
        pretrained_backbone: Use ImageNet pre-trained weights.
        freeze_backbone: Freeze the semantic backbone (linear-probe mode).
    """

    def __init__(
        self,
        num_classes: int = 38,
        backbone_name: str = "vit_base_patch16_224",
        fusion_dim: int = 256,
        pc_scales: int = 4,
        pc_orientations: int = 6,
        image_size: tuple[int, int] = (224, 224),
        num_heads: int = 4,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.image_size = image_size

        # --- Stream 1: Phase Congruency ---
        self.filter_bank = LogGaborFilterBank(
            image_size=image_size,
            num_scales=pc_scales,
            num_orientations=pc_orientations,
        )
        self.pc_extractor = PhaseCongruencyExtractor(
            num_scales=pc_scales,
            num_orientations=pc_orientations,
        )
        self.pc_encoder = PCEncoder(
            in_channels=3,  # magnitude + symmetry + oriented_energy
            fusion_dim=fusion_dim,
        )

        # --- Stream 2: Semantic Backbone (ViT) ---
        self.backbone = SemanticBackbone(
            backbone_name=backbone_name,
            fusion_dim=fusion_dim,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )

        # --- Stream 3: Illumination Normalization ---
        self.illum_stream = IlluminationNormStream(
            out_dim=fusion_dim,
        )

        # --- Cross-Attention Fusion ---
        self.fusion = StructuralSemanticFusion(
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # --- Classification Head ---
        # Input: fused (fusion_dim) + illumination auxiliary (fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to grayscale using standard luminance weights.

        Args:
            x: (B, 3, H, W) RGB tensor.

        Returns:
            (B, 1, H, W) grayscale tensor.
        """
        # ITU-R BT.601 luminance
        weights = torch.tensor(
            [0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_clahe: torch.Tensor | None = None,
        return_maps: bool = False,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through all three streams + fusion.

        Args:
            x_rgb: (B, 3, H, W) raw RGB input.
            x_clahe: (B, 3, H, W) CLAHE-preprocessed input.  If None,
                uses ``x_rgb`` directly (CLAHE should be in data pipeline).
            return_maps: If True, include PC maps in output dict.
            return_attention: If True, include attention weights.

        Returns:
            Dict with at minimum ``'logits': (B, num_classes)``.
            Optionally ``'pc_maps'``, ``'attn_weights'``, ``'fused'``.
        """
        if x_clahe is None:
            x_clahe = x_rgb

        # --- Stream 1: Phase Congruency ---
        gray = self._rgb_to_gray(x_rgb)  # (B, 1, H, W)
        even, odd = self.filter_bank(gray)  # each (B, 24, H, W)
        pc_maps = self.pc_extractor(even, odd)  # dict of (B, 1, H, W) maps

        # Concatenate 3 PC maps -> (B, 3, H, W)
        pc_input = torch.cat(
            [pc_maps["pc_magnitude"], pc_maps["phase_symmetry"], pc_maps["oriented_energy"]],
            dim=1,
        )
        structural_tokens = self.pc_encoder(pc_input)  # (B, 49, fusion_dim)

        # --- Stream 2: Semantic Backbone ---
        semantic_tokens = self.backbone(x_rgb)  # (B, 196, fusion_dim) for ViT-B/16

        # --- Stream 3: Illumination Normalization ---
        illum_features = self.illum_stream(x_clahe)  # (B, fusion_dim)

        # --- Fusion ---
        fused, attn_weights = self.fusion(
            structural_tokens, semantic_tokens, return_attention=return_attention
        )  # fused: (B, fusion_dim)

        # --- Classification ---
        combined = torch.cat([fused, illum_features], dim=1)  # (B, fusion_dim * 2)
        logits = self.classifier(combined)  # (B, num_classes)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if return_maps:
            output["pc_maps"] = pc_input  # (B, 3, H, W)
            output["pc_magnitude"] = pc_maps["pc_magnitude"]
            output["phase_symmetry"] = pc_maps["phase_symmetry"]
            output["oriented_energy"] = pc_maps["oriented_energy"]
        if return_attention and attn_weights is not None:
            output["attn_weights"] = attn_weights
        if return_maps:
            output["fused"] = fused

        return output

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per sub-module."""
        counts = {}
        for name, module in [
            ("filter_bank", self.filter_bank),
            ("pc_extractor", self.pc_extractor),
            ("pc_encoder", self.pc_encoder),
            ("backbone", self.backbone),
            ("illum_stream", self.illum_stream),
            ("fusion", self.fusion),
            ("classifier", self.classifier),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
