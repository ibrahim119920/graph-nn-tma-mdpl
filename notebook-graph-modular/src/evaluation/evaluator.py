"""Teacher-forced validation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import mae, per_node_rmse, rmse


@dataclass(frozen=True)
class EvaluationResult:
    prediction_normalized: np.ndarray
    target_normalized: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    rmse: float
    mae: float
    per_node_rmse: np.ndarray


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    denormalize_target: Callable[[np.ndarray], np.ndarray],
) -> EvaluationResult:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device)
            prediction = model(feature_batch).cpu().numpy()
            predictions.append(prediction)
            targets.append(target_batch.numpy())

    if not predictions:
        raise ValueError("DataLoader evaluation kosong; tidak ada output untuk divalidasi.")
    prediction_normalized = np.concatenate(predictions, axis=0)
    target_normalized = np.concatenate(targets, axis=0)
    if not np.isfinite(prediction_normalized).all():
        raise FloatingPointError("Output model evaluation mengandung NaN atau infinity.")
    prediction = denormalize_target(prediction_normalized)
    target = denormalize_target(target_normalized)
    return EvaluationResult(
        prediction_normalized=prediction_normalized,
        target_normalized=target_normalized,
        prediction=prediction,
        target=target,
        rmse=rmse(prediction, target),
        mae=mae(prediction, target),
        per_node_rmse=per_node_rmse(prediction, target),
    )
