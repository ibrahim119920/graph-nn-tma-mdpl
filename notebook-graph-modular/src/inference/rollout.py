"""Causal water-feature reconstruction and autoregressive inference."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AutoregressiveFeatureSpec:
    water_level_position: int
    lag_positions: list[int]
    rolling_positions: dict[int, dict[str, int]]

    @classmethod
    def from_feature_columns(
        cls,
        feature_columns: list[str],
        n_lags: int,
    ) -> "AutoregressiveFeatureSpec":
        water_level_position = feature_columns.index("wl_t")
        lag_positions = [
            feature_columns.index(f"wl_t-{lag}")
            for lag in range(1, n_lags + 1)
        ]
        rolling_windows = sorted(
            {
                int(match.group(1))
                for column in feature_columns
                if (
                    match := re.match(
                        r"wl_roll(\d+)_(mean|std|min|max)$", column
                    )
                )
            }
        )
        rolling_positions = {
            window: {
                statistic: feature_columns.index(
                    f"wl_roll{window}_{statistic}"
                )
                for statistic in ("mean", "std", "min", "max")
            }
            for window in rolling_windows
        }
        return cls(
            water_level_position=water_level_position,
            lag_positions=lag_positions,
            rolling_positions=rolling_positions,
        )


@dataclass(frozen=True)
class InferenceResult:
    predictions: dict[pd.Timestamp, np.ndarray]
    water_level_series: np.ndarray
    skipped_datetimes: list[pd.Timestamp]
    filled_training_cells: int
    feature_spec: AutoregressiveFeatureSpec
    fallback_count: int = 0


def rebuild_water_window(
    window: np.ndarray,
    start_position: int,
    water_level_series: np.ndarray,
    feature_spec: AutoregressiveFeatureSpec,
) -> np.ndarray:
    for offset, position in enumerate(
        range(start_position, start_position + window.shape[0])
    ):
        window[
            offset, :, feature_spec.water_level_position
        ] = water_level_series[position]
        for lag, lag_position in enumerate(
            feature_spec.lag_positions, start=1
        ):
            source_position = position - lag
            if source_position >= 0:
                window[offset, :, lag_position] = water_level_series[
                    source_position
                ]
        for rolling_window, columns in (
            feature_spec.rolling_positions.items()
        ):
            rolling_start = position - rolling_window
            if rolling_start >= 0:
                values = water_level_series[rolling_start:position]
                window[offset, :, columns["mean"]] = values.mean(axis=0)
                window[offset, :, columns["std"]] = values.std(
                    axis=0, ddof=1
                )
                window[offset, :, columns["min"]] = values.min(axis=0)
                window[offset, :, columns["max"]] = values.max(axis=0)
    return window


def normalize_inference_window(
    window: np.ndarray,
    station_center: np.ndarray,
    station_scale: np.ndarray,
    water_value_positions: list[int],
    water_std_positions: list[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    result = window.copy()
    num_nodes = station_center.shape[0]
    center = station_center.reshape(1, 1, num_nodes)
    scale = station_scale.reshape(1, 1, num_nodes)
    for position in water_value_positions:
        result[..., position] = (result[..., position] - center) / scale
    for position in water_std_positions:
        result[..., position] = result[..., position] / scale
    return (result - feature_mean) / feature_std


def predict_autoregressive(
    model: nn.Module,
    panel: np.ndarray,
    observation_datetimes: pd.DatetimeIndex,
    target_datetimes: pd.DatetimeIndex,
    feature_columns: list[str],
    *,
    train_end: pd.Timestamp | str,
    n_lags: int,
    time_window: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    station_center: np.ndarray,
    station_scale: np.ndarray,
    station_low: np.ndarray,
    station_high: np.ndarray,
    water_value_positions: list[int],
    water_std_positions: list[int],
    device: torch.device | str,
) -> InferenceResult:
    feature_spec = AutoregressiveFeatureSpec.from_feature_columns(
        feature_columns, n_lags
    )
    if target_datetimes.empty:
        raise ValueError("Tidak ada datetime target pada sample_submission.")
    if panel.ndim != 3:
        raise ValueError(
            f"panel inference harus 3D (time,nodes,features), diterima: {panel.shape}"
        )
    if panel.shape[1] != len(station_center) or panel.shape[2] != len(feature_columns):
        raise ValueError(
            "Kontrak panel inference tidak cocok dengan checkpoint: "
            f"panel={panel.shape}, nodes/features checkpoint="
            f"{len(station_center)}/{len(feature_columns)}."
        )
    datetime_to_position = {
        dt: i for i, dt in enumerate(observation_datetimes)
    }
    water_level_series = panel[
        :, :, feature_spec.water_level_position
    ].copy()

    train_end_timestamp = pd.Timestamp(train_end)
    if train_end_timestamp not in datetime_to_position:
        raise ValueError(
            f"train_end {train_end_timestamp} tidak ada pada obs_datetimes dataset."
        )
    missing_targets = target_datetimes.difference(observation_datetimes)
    if len(missing_targets):
        raise ValueError(
            "Datetime sample_submission tidak ada pada observasi dataset: "
            f"{missing_targets[:3].tolist()}"
        )
    last_train_position = datetime_to_position[train_end_timestamp]
    filled_training_cells = 0
    for node_index in range(station_center.shape[0]):
        column = water_level_series[
            : last_train_position + 1, node_index
        ]
        if np.isnan(column).any():
            filled_training_cells += int(np.isnan(column).sum())
            water_level_series[
                : last_train_position + 1, node_index
            ] = pd.Series(column).ffill().values

    first_target_position = datetime_to_position[target_datetimes.min()]
    first_window_start = first_target_position - time_window
    maximum_rolling_window = max(feature_spec.rolling_positions, default=0)
    history_start = max(
        0,
        first_window_start - max(n_lags, maximum_rolling_window),
    )
    required_history = water_level_series[
        history_start:first_target_position
    ]
    if np.isnan(required_history).any() or not np.isfinite(required_history).all():
        raise ValueError(
            "Histori tinggi air di batas train -> test kosong/non-finite; "
            "periksa data train, jangan mengisi dengan informasi masa depan."
        )

    predictions: dict[pd.Timestamp, np.ndarray] = {}
    skipped: list[pd.Timestamp] = []
    model.eval()
    with torch.no_grad():
        for target_datetime in target_datetimes:
            if target_datetime not in datetime_to_position:
                skipped.append(target_datetime)
                continue

            target_position = datetime_to_position[target_datetime]
            end_position = target_position - 1
            start_position = end_position - (time_window - 1)
            if end_position < 0 or start_position < 0:
                skipped.append(target_datetime)
                continue

            window = panel[start_position : end_position + 1].copy()
            window = rebuild_water_window(
                window,
                start_position,
                water_level_series,
                feature_spec,
            )
            if np.isnan(window).any():
                skipped.append(target_datetime)
                continue

            normalized = normalize_inference_window(
                window[None],
                station_center,
                station_scale,
                water_value_positions,
                water_std_positions,
                feature_mean,
                feature_std,
            )[0]
            feature_batch = torch.from_numpy(
                normalized[None].astype(np.float32)
            ).to(device)
            prediction_normalized = model(feature_batch).cpu().numpy()[0]
            if not np.isfinite(prediction_normalized).all():
                raise FloatingPointError(
                    f"Output model non-finite pada datetime {target_datetime}."
                )
            raw_prediction = (
                prediction_normalized * station_scale + station_center
            )
            prediction = np.clip(
                raw_prediction, station_low, station_high
            )
            predictions[target_datetime] = prediction
            water_level_series[target_position] = prediction

    return InferenceResult(
        predictions=predictions,
        water_level_series=water_level_series,
        skipped_datetimes=skipped,
        filled_training_cells=filled_training_cells,
        feature_spec=feature_spec,
    )
