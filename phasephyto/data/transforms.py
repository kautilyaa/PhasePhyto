"""
Image transforms and augmentation pipelines for PhasePhyto.

Provides separate train/val/test transforms per use case, plus the
CLAHE preprocessing utility used by the illumination-normalised stream.
"""

from typing import Any, cast

import cv2
import numpy as np
import torch
from torchvision import transforms


class CLAHETransform:
    """Torchvision-compatible transform: applies CLAHE to PIL/numpy images.

    Applied to the L channel of CIELAB colour space, preserving A and B
    colour vectors.  Returns a second tensor for the illumination stream.

    Args:
        clip_limit: CLAHE contrast clip limit.
        tile_grid_size: Grid dimensions for local histogram equalisation.
    """

    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, image: Any) -> np.ndarray:
        """Apply CLAHE.

        Args:
            image: PIL Image or numpy (H, W, 3) uint8 RGB.

        Returns:
            (H, W, 3) uint8 RGB with luminance equalised.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self.clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cast(np.ndarray, cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class DualTransform:
    """Wraps standard transforms to produce both RGB and CLAHE tensors.

    Returns a tuple ``(rgb_tensor, clahe_tensor)`` so the dataloader
    can feed both streams of PhasePhyto.

    Args:
        base_transform: Spatial augmentations (resize, crop, flip) applied
            to both RGB and CLAHE identically.
        normalize: Normalisation applied to both tensors.
        clahe_transform: The CLAHETransform instance.
    """

    def __init__(
        self,
        base_transform: transforms.Compose,
        normalize: transforms.Normalize,
        clahe_transform: CLAHETransform | None = None,
    ):
        self.base = base_transform
        self.normalize = normalize
        self.clahe_fn = clahe_transform or CLAHETransform()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply spatial augmentations (returns PIL)
        augmented = self.base(image)

        # Convert to numpy for CLAHE
        aug_np = np.array(augmented)
        clahe_np = self.clahe_fn(aug_np)

        # To tensor + normalize
        rgb_tensor = self.normalize(self.to_tensor(aug_np))
        clahe_tensor = self.normalize(self.to_tensor(clahe_np))

        return rgb_tensor, clahe_tensor


# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> DualTransform:
    """Training augmentations with random crop, flip, colour jitter."""
    spatial = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ])
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return DualTransform(spatial, normalize)


def get_val_transforms(image_size: int = 224) -> DualTransform:
    """Validation/test transforms: deterministic resize + center crop."""
    spatial = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
    ])
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return DualTransform(spatial, normalize)


def clahe_preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Standalone CLAHE preprocessing function for inference scripts."""
    return CLAHETransform()(image_rgb)
