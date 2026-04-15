"""
Semantic backbone stream: extracts high-level features from raw RGB images.

Wraps any ``timm`` model and projects its intermediate features to the
fusion dimension expected by the cross-attention module.  Default backbone
is ViT-B/16 (``vit_base_patch16_224``).

For ViT: patch tokens (196 for 224x224 / 16x16) serve as semantic tokens.
For CNN:  feature map is flattened spatially into a token sequence.
"""

from typing import Any, cast

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None  # type: ignore[assignment]


class SemanticBackbone(nn.Module):
    """Pre-trained backbone with projection to fusion dimension.

    Args:
        backbone_name: Any ``timm`` model name.  ViT variants recommended:
            ``'vit_base_patch16_224'``, ``'vit_small_patch16_224'``,
            ``'vit_large_patch16_224'``.  CNN backbones also supported:
            ``'efficientnet_b0'``, ``'convnext_tiny'``, ``'resnet50'``.
        fusion_dim: Output embedding dimension per token.
        pretrained: Use ImageNet pre-trained weights.
        freeze_backbone: Freeze all backbone parameters (linear-probe mode).
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        fusion_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required: pip install timm")

        self.backbone_name = backbone_name
        self.fusion_dim = fusion_dim
        self._is_vit = "vit" in backbone_name.lower()

        if self._is_vit:
            # Create ViT without classification head
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0
            )
            # ViT embed_dim (e.g. 768 for vit_base)
            backbone_dim = int(cast(Any, self.backbone).embed_dim)
        else:
            # CNN backbone: extract features before global pool
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
            )
            # Infer output channels by a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.backbone(dummy)
                backbone_dim = out.shape[1] if out.dim() == 4 else out.shape[-1]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 1x1 projection to fusion dimension
        self.proj = nn.Linear(backbone_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract semantic tokens from RGB input.

        Args:
            x: (B, 3, 224, 224) RGB image tensor.

        Returns:
            (B, num_tokens, fusion_dim) semantic token sequence.
            For ViT-B/16: num_tokens = 196 (14x14 patch grid).
            For CNNs: num_tokens = spatial_h * spatial_w.
        """
        if self._is_vit:
            # timm ViT with num_classes=0 returns (B, num_patches, embed_dim)
            # via forward_features -> removes CLS token in recent timm versions
            tokens = cast(Any, self.backbone).forward_features(x)
            # Some timm versions include CLS token at position 0
            if hasattr(self.backbone, "num_prefix_tokens"):
                n_prefix = int(cast(Any, self.backbone).num_prefix_tokens)
                if n_prefix > 0:
                    tokens = tokens[:, n_prefix:, :]  # drop CLS / register tokens
        else:
            # CNN: (B, C, H, W) -> flatten spatial -> (B, H*W, C)
            features = self.backbone(x)  # (B, C, H, W)
            B, C, Hf, Wf = features.shape
            tokens = features.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project to fusion dimension: (B, num_tokens, backbone_dim) -> (B, num_tokens, fusion_dim)
        projected = self.proj(tokens)
        return cast(torch.Tensor, self.norm(projected))
