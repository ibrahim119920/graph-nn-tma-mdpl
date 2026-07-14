"""NumPy metrics matching the formulas in the notebook."""

from __future__ import annotations

import numpy as np


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def per_node_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    return np.sqrt(np.mean((prediction - target) ** 2, axis=0))


def normalized_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    station_scale: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(((prediction - target) / station_scale) ** 2)
        )
    )

