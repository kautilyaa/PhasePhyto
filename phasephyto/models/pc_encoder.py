"""
Lightweight CNN encoder that compresses 3-channel PC maps into Structural Tokens.

Input:  concatenated (pc_magnitude, phase_symmetry, oriented_energy) -> (B, 3, H, W)
Output: structural token sequence -> (B, num_tokens, fusion_dim)

The spatial output is 7x7 = 49 tokens by default, matching common ViT/CNN feature
grid dimensions for cross-attention compatibility.
"""

from typing import cast

import torch
import torch.nn as nn


class PCEncoder(nn.Module):
    """Two-layer CNN that converts PC maps to structural tokens.

    Args:
        in_channels: Number of input PC map channels (default 3: magnitude,
            symmetry, oriented energy).
        mid_channels: Hidden channel dimension.
        fusion_dim: Output token embedding dimension (must match semantic tokens).
        spatial_size: Spatial grid size for adaptive pooling (tokens = size^2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        fusion_dim: int = 256,
        spatial_size: int = 7,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.fusion_dim = fusion_dim

        self.encoder = nn.Sequential(
            # Layer 1: (B, 3, H, W) -> (B, mid, H/2, W/2)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            # Layer 2: (B, mid, H/2, W/2) -> (B, fusion_dim, H/4, W/4)
            nn.Conv2d(mid_channels, fusion_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
            # Pool to fixed spatial size
            nn.AdaptiveAvgPool2d((spatial_size, spatial_size)),
        )

    @property
    def num_tokens(self) -> int:
        return self.spatial_size**2

    def forward(self, pc_maps: torch.Tensor) -> torch.Tensor:
        """Encode PC maps into structural tokens.

        Args:
            pc_maps: (B, 3, H, W) concatenated PC maps.

        Returns:
            (B, spatial_size^2, fusion_dim) structural token sequence.
        """
        # (B, 3, H, W) -> (B, fusion_dim, spatial_size, spatial_size)
        features = self.encoder(pc_maps)
        B, C, Hs, Ws = features.shape  # (B, fusion_dim, 7, 7)

        # Flatten spatial dims and transpose to token sequence
        # (B, C, Hs, Ws) -> (B, C, Hs*Ws) -> (B, Hs*Ws, C)
        tokens = features.flatten(2).transpose(1, 2)  # (B, 49, fusion_dim)
        return cast(torch.Tensor, tokens)
