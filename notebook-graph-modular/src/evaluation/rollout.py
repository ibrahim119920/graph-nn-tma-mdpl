"""Leakage-safe autoregressive rollout validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.inference.rollout import (
    AutoregressiveFeatureSpec,
    rebuild_water_window,
)
from src.models.utils import unwrap_model

from .metrics import normalized_rmse


@dataclass(frozen=True)
class RolloutValidationContext:
    panel: np.ndarray
    observation_datetimes: pd.DatetimeIndex
    training_datetimes: pd.DatetimeIndex
    validation_start_index: int
    time_window: int
    feature_spec: AutoregressiveFeatureSpec
    station_center: np.ndarray
    station_scale: np.ndarray
    station_low: np.ndarray
    station_high: np.ndarray
    normalize_features: Callable[[np.ndarray], np.ndarray]


def evaluate_rollout(
    model: nn.Module,
    context: RolloutValidationContext,
    device: torch.device | str,
) -> float:
    base_model = unwrap_model(model)
    base_model.eval()
    actual_water = context.panel[
        :, :, context.feature_spec.water_level_position
    ].copy()
    rollout_water = actual_water.copy()
    validation_datetimes = pd.DatetimeIndex(
        context.training_datetimes[context.validation_start_index :]
    )
    datetime_to_position = {
        dt: i for i, dt in enumerate(context.observation_datetimes)
    }
    validation_input_positions = [
        datetime_to_position[dt] for dt in validation_datetimes
    ]
    metric_input_positions = set(validation_input_positions)
    first_input_position = validation_input_positions[0]
    last_input_position = validation_input_positions[-1]
    rollout_water[first_input_position + 1 :] = np.nan
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        # Prediksi setiap timestep perantara. Ini mencegah gap pada daftar
        # sample valid diisi ulang memakai observasi aktual validation.
        for input_position in range(
            first_input_position, last_input_position + 1
        ):
            target_position = input_position + 1
            if target_position >= len(context.observation_datetimes):
                continue
            start = input_position - (context.time_window - 1)
            if start < 0:
                continue
            window = context.panel[start : input_position + 1].copy()
            window = rebuild_water_window(
                window,
                start,
                rollout_water,
                context.feature_spec,
            )
            if np.isnan(window).any():
                return float("inf")
            feature_batch = torch.from_numpy(
                context.normalize_features(window[None]).astype(np.float32)
            ).to(device)
            prediction_normalized = base_model(feature_batch).cpu().numpy()[0]
            prediction = (
                prediction_normalized * context.station_scale
                + context.station_center
            )
            prediction = np.clip(
                prediction, context.station_low, context.station_high
            )
            rollout_water[target_position] = prediction
            if (
                input_position in metric_input_positions
                and not np.isnan(actual_water[target_position]).any()
            ):
                predictions.append(prediction)
                targets.append(actual_water[target_position])

    if not predictions:
        return float("inf")
    return normalized_rmse(
        np.stack(predictions),
        np.stack(targets),
        context.station_scale,
    )
