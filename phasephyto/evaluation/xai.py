"""Explainability tools for PhasePhyto: Grad-CAM and attention visualisation."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure


class GradCAMPhasePhyto:
    """Grad-CAM for the cross-attention layer of PhasePhyto.

    Hooks into the cross-attention output to compute class-discriminative
    spatial heat maps that show *where* structural tokens attend.

    Args:
        model: Trained PhasePhyto model.
        target_layer: The module to hook (default: fusion cross-attention).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or cast(Any, model).fusion.cross_attn
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)  # type: ignore[arg-type]
        self.target_layer.register_full_backward_hook(self._save_gradient)  # type: ignore[arg-type]

    def _save_activation(self, module: nn.Module, input: tuple, output: tuple) -> None:
        self.activations = output[0].detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        x_rgb: torch.Tensor,
        x_clahe: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap.

        Args:
            x_rgb: (1, 3, H, W) single image tensor.
            x_clahe: (1, 3, H, W) CLAHE version (optional).
            target_class: Class index for gradient computation.
                If None, uses the predicted class.

        Returns:
            (H, W) numpy heatmap normalised to [0, 1].
        """
        self.model.eval()
        x_rgb.requires_grad_(True)

        output = self.model(x_rgb, x_clahe=x_clahe)
        logits = output["logits"]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        # Weight activations by gradients
        weights = self.gradients.mean(dim=1, keepdim=True)  # (1, 1, D)
        cam = (weights * self.activations).sum(dim=-1)  # (1, Nq)
        cam = torch.relu(cam)

        # Reshape to spatial grid (7x7 for 49 tokens)
        side = int(cam.shape[1] ** 0.5)
        cam = cam.view(1, 1, side, side)

        # Upsample to image resolution
        cam = torch.nn.functional.interpolate(
            cam, size=x_rgb.shape[2:], mode="bilinear", align_corners=False
        )
        cam_np = cam.squeeze().cpu().numpy()

        # Normalise
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cast(np.ndarray, cam_np)


def visualize_attention(
    image_rgb: np.ndarray,
    pc_maps: dict[str, np.ndarray],
    grad_cam: np.ndarray | None = None,
    prediction: str = "",
    save_path: str | None = None,
) -> Figure:
    """Create a multi-panel visualisation figure.

    Panels: Original | PC Magnitude | Phase Symmetry | Oriented Energy | Grad-CAM

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        pc_maps: Dict with ``'pc_magnitude'``, ``'phase_symmetry'``,
            ``'oriented_energy'`` as (H, W) float arrays.
        grad_cam: (H, W) Grad-CAM heatmap (optional).
        prediction: Predicted class label string.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    n_panels = 4 + (1 if grad_cam is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Input\n{prediction}")
    axes[0].axis("off")

    for i, (key, cmap) in enumerate([
        ("pc_magnitude", "hot"),
        ("phase_symmetry", "magma"),
        ("oriented_energy", "viridis"),
    ]):
        axes[i + 1].imshow(pc_maps[key], cmap=cmap)
        axes[i + 1].set_title(key.replace("_", " ").title())
        axes[i + 1].axis("off")

    if grad_cam is not None:
        axes[4].imshow(image_rgb)
        axes[4].imshow(grad_cam, cmap="jet", alpha=0.5)
        axes[4].set_title("Grad-CAM")
        axes[4].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
