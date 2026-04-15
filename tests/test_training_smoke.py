"""Tiny end-to-end training smoke test on synthetic tensors."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from phasephyto.models import PhasePhyto, TimmClassifier
from phasephyto.training.trainer import Trainer


def test_tiny_synthetic_training_smoke(tmp_path: Path):
    """One epoch should train and validate without data files or downloads."""
    torch.manual_seed(7)
    rgb = torch.rand(4, 3, 32, 32)
    clahe = rgb.clone()
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loader = DataLoader(TensorDataset(rgb, clahe, labels), batch_size=2, shuffle=False)
    model = PhasePhyto(
        num_classes=2,
        backbone_name="resnet18",
        fusion_dim=32,
        pc_scales=2,
        pc_orientations=4,
        image_size=(32, 32),
        num_heads=4,
        dropout=0.0,
        pretrained_backbone=False,
    )

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        lr=1e-4,
        epochs=1,
        warmup_epochs=1,
        checkpoint_dir=tmp_path,
        device="cpu",
    )
    history = trainer.fit()

    assert len(history["train_loss"]) == 1
    assert len(history["val_f1"]) == 1
    assert history["train_loss"][0] >= 0.0


def test_baseline_output_contract():
    """Semantic baseline should be usable by the shared trainer/evaluator."""
    model = TimmClassifier(num_classes=2, backbone_name="resnet18", pretrained=False)
    output = model(torch.rand(2, 3, 32, 32), x_clahe=torch.rand(2, 3, 32, 32))

    assert output["logits"].shape == (2, 2)
