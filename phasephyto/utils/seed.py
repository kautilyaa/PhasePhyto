"""Reproducibility utilities."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python stdlib, NumPy, PyTorch CPU, PyTorch CUDA,
    and configures cuDNN for deterministic behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
