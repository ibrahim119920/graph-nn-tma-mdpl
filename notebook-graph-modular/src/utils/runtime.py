"""Runtime helpers for reproducible, CPU/GPU-safe execution."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device() -> torch.device:
    """Use CUDA only when it is actually available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed every random source used by this project.

    Deterministic PyTorch algorithms are deliberately not forced because some
    graph/CUDA kernels may not support them. CPU and a fixed software stack are
    reproducible to the usual PyTorch tolerance; GPU results can still vary
    slightly between CUDA, driver, and hardware versions.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"Seed harus bilangan bulat non-negatif, diterima: {seed!r}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
