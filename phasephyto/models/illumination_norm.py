"""
Illumination-Normalized Stream: CIELAB colour space + CLAHE on luminance.

Converts RGB -> CIELAB, applies Contrast Limited Adaptive Histogram
Equalisation (CLAHE) to the L channel only (preserving true colour vectors
in A and B), then feeds the result through a shallow CNN to produce an
auxiliary semantic feature vector.

CLAHE is non-differentiable and is applied as a pre-processing step
(not inside the autograd graph).  The shallow CNN that follows *is*
differentiable.
"""

from typing import cast

import cv2
import numpy as np
import torch
import torch.nn as nn


class CLAHEPreprocessor:
    """Applies CLAHE to the L channel of a CIELAB image (numpy/OpenCV).

    This is a callable transform, not an nn.Module, because CLAHE is
    non-differentiable.  Use it in your data pipeline or call it before
    the forward pass.

    Args:
        clip_limit: CLAHE contrast clip limit.
        tile_grid_size: Grid size for local histogram equalisation.
    """

    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Apply CLAHE on L channel, return RGB uint8.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            (H, W, 3) uint8 RGB image with luminance-equalised.
        """
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class IlluminationNormStream(nn.Module):
    """Shallow CNN that processes CLAHE-normalised images.

    Expects images that have *already* been processed by ``CLAHEPreprocessor``
    (in the data transform pipeline).  Outputs a global auxiliary feature
    vector that is concatenated with the cross-attention output before the
    classification head.

    Args:
        in_channels: Input channels (3 for RGB).
        mid_channels: Hidden layer channels.
        out_dim: Output feature vector dimension (should match fusion_dim).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        out_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x_clahe: torch.Tensor) -> torch.Tensor:
        """Extract auxiliary illumination-normalised features.

        Args:
            x_clahe: (B, 3, H, W) CLAHE-preprocessed RGB tensor.

        Returns:
            (B, out_dim) feature vector.
        """
        return cast(torch.Tensor, self.net(x_clahe).flatten(1))  # (B, out_dim)
