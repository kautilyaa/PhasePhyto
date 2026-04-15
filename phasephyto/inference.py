"""Single-image or batch inference with PC map visualisation."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from phasephyto.data.transforms import IMAGENET_MEAN, IMAGENET_STD, CLAHETransform
from phasephyto.evaluation.xai import GradCAMPhasePhyto, visualize_attention
from phasephyto.models import PhasePhyto
from phasephyto.utils.config import load_config


def preprocess(image_path: str, image_size: int = 224):
    """Load and preprocess a single image, returning RGB + CLAHE tensors."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    rgb_tensor = tf(img).unsqueeze(0)  # (1, 3, H, W)

    clahe_np = CLAHETransform()(cv2.resize(img_np, (image_size, image_size)))
    clahe_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(clahe_np).unsqueeze(0)

    # Also return display-ready resized image
    display_img = cv2.resize(img_np, (image_size, image_size))
    return rgb_tensor, clahe_tensor, display_img


def main():
    parser = argparse.ArgumentParser(description="PhasePhyto Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--class-names", type=str, nargs="*", help="Class label names")
    parser.add_argument("--output-dir", type=str, default="inference_output")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlays")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    image_size = cfg.data.image_size

    num_classes = len(args.class_names) if args.class_names else 38
    model = PhasePhyto(
        num_classes=num_classes,
        backbone_name=cfg.model.backbone_name,
        fusion_dim=cfg.model.fusion_dim,
        image_size=(image_size, image_size),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    gradcam = GradCAMPhasePhyto(model) if args.gradcam else None

    input_path = Path(args.input)
    image_paths = list(input_path.glob("*")) if input_path.is_dir() else [input_path]
    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        rgb, clahe, display = preprocess(str(img_path), image_size)
        rgb, clahe = rgb.to(device), clahe.to(device)

        with torch.no_grad():
            output = model(rgb, x_clahe=clahe, return_maps=True)

        pred_idx = output["logits"].argmax(dim=1).item()
        pred_label = args.class_names[pred_idx] if args.class_names else str(pred_idx)
        confidence = torch.softmax(output["logits"], dim=1).max().item()

        print(f"{img_path.name}: {pred_label} ({confidence:.2%})")

        # Extract PC maps for visualisation
        pc_maps = {
            "pc_magnitude": output["pc_magnitude"][0, 0].cpu().numpy(),
            "phase_symmetry": output["phase_symmetry"][0, 0].cpu().numpy(),
            "oriented_energy": output["oriented_energy"][0, 0].cpu().numpy(),
        }

        cam = None
        if gradcam is not None:
            rgb_grad = rgb.clone().requires_grad_(True)
            cam = gradcam(rgb_grad, clahe)

        fig = visualize_attention(
            display, pc_maps, grad_cam=cam,
            prediction=f"{pred_label} ({confidence:.1%})",
            save_path=str(output_dir / f"{img_path.stem}_analysis.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
