"""
Training loop for PhasePhyto with mixed precision, gradient clipping,
cosine annealing, early stopping, and optional wandb logging.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    """PhasePhyto training manager.

    Args:
        model: The PhasePhyto model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        lr: Base learning rate.
        weight_decay: AdamW weight decay.
        epochs: Maximum training epochs.
        warmup_epochs: Linear warmup epochs.
        grad_clip: Max gradient norm for clipping.
        patience: Early stopping patience (0 = disabled).
        checkpoint_dir: Directory for saving checkpoints.
        device: ``'cuda'`` or ``'cpu'``.
        use_wandb: Enable wandb logging.
        project_name: wandb project name.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        epochs: int = 50,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        patience: int = 10,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "cuda",
        use_wandb: bool = False,
        project_name: str = "phasephyto",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb and HAS_WANDB

        # Optimiser
        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler: cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(epochs - warmup_epochs, 1), T_mult=1
        )

        # Mixed precision
        amp_device = device.split(":")[0]
        self.scaler = GradScaler(amp_device, enabled=(amp_device == "cuda"))

        # Tracking
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0

        if self.use_wandb:
            wandb.init(project=project_name, config={
                "lr": lr, "weight_decay": weight_decay, "epochs": epochs,
                "warmup_epochs": warmup_epochs, "grad_clip": grad_clip,
            })

    def _warmup_lr(self, epoch: int) -> None:
        """Linear warmup for the first few epochs."""
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * factor if "initial_lr" in pg else pg["lr"]

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]")
        for batch in pbar:
            # Unpack: supports both (rgb, label) and (rgb, clahe, label)
            if len(batch) == 3:
                rgb, clahe, labels = batch
                rgb = rgb.to(self.device)
                clahe = clahe.to(self.device)
                labels = labels.to(self.device)
            else:
                rgb, labels = batch
                rgb, labels = rgb.to(self.device), labels.to(self.device)
                clahe = None

            self.optimizer.zero_grad()

            with autocast(self.device.split(":")[0], enabled=(self.device != "cpu")):
                output = self.model(rgb, x_clahe=clahe)
                logits = output["logits"]
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        return {
            "train_loss": total_loss / total,
            "train_acc": correct / total,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in tqdm(self.val_loader, desc="[val]"):
            if len(batch) == 3:
                rgb, clahe, labels = batch
                rgb = rgb.to(self.device)
                clahe = clahe.to(self.device)
                labels = labels.to(self.device)
            else:
                rgb, labels = batch
                rgb, labels = rgb.to(self.device), labels.to(self.device)
                clahe = None

            output = self.model(rgb, x_clahe=clahe)
            logits = output["logits"]
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        total = len(all_labels)
        acc = sum(
            pred == label for pred, label in zip(all_preds, all_labels, strict=True)
        ) / total

        # Macro F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "val_loss": total_loss / total,
            "val_acc": acc,
            "val_f1": f1,
        }

    def fit(self) -> dict[str, Any]:
        """Full training loop with early stopping."""
        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
        }

        for epoch in range(self.epochs):
            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            # Step scheduler after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Log
            for k, v in {**train_metrics, **val_metrics}.items():
                history.setdefault(k, []).append(v)

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}: "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"val_acc={val_metrics['val_acc']:.4f} "
                f"val_f1={val_metrics['val_f1']:.4f} "
                f"lr={lr:.6f}"
            )

            if self.use_wandb:
                wandb.log({**train_metrics, **val_metrics, "lr": lr, "epoch": epoch})

            # Checkpoint best model
            if val_metrics["val_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_f1"]
                self.epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": self.best_val_f1,
                }, self.checkpoint_dir / "best_model.pt")
                print(f"  -> Saved best model (F1={self.best_val_f1:.4f})")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.patience > 0 and self.epochs_no_improve >= self.patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        if self.use_wandb:
            wandb.finish()

        return history
