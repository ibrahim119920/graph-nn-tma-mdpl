"""Feature-panel construction and chronological train/validation splitting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def chronological_split_sizes(
    sample_count: int,
    validation_ratio: float = 0.1,
) -> tuple[int, int]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio harus berada di antara 0 dan 1.")
    if sample_count < 2:
        raise ValueError("Minimal dua sample diperlukan untuk train/validation split.")
    n_validation = max(1, int(sample_count * validation_ratio))
    return sample_count - n_validation, n_validation


def find_complete_timesteps(
    node_features: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    required_columns: list[str],
) -> pd.DatetimeIndex:
    period_mask = node_features["datetime"].between(start, end, inclusive="both")
    completeness = (
        node_features.loc[period_mask]
        .set_index(["datetime", "nama_pos"])[required_columns]
        .notna()
        .all(axis=1)
        .groupby(level=0)
        .all()
    )
    return pd.DatetimeIndex(sorted(completeness[completeness].index))


def build_feature_panel(
    node_features: pd.DataFrame,
    observation_datetimes: pd.DatetimeIndex,
    node_order: list[str],
    feature_columns: list[str],
    target_column: str = "y_next",
) -> tuple[np.ndarray, np.ndarray, dict[pd.Timestamp, int]]:
    full_index = pd.MultiIndex.from_product(
        [observation_datetimes, node_order],
        names=["datetime", "nama_pos"],
    )
    panel = (
        node_features.set_index(["datetime", "nama_pos"])[feature_columns]
        .reindex(full_index)
    )
    panel_arr = panel.values.reshape(
        len(observation_datetimes), len(node_order), len(feature_columns)
    ).astype(np.float32)

    target_series = (
        node_features.set_index(["datetime", "nama_pos"])[target_column]
        .reindex(full_index)
    )
    target_arr = target_series.values.reshape(
        len(observation_datetimes), len(node_order)
    ).astype(np.float32)
    datetime_to_position = {
        dt: i for i, dt in enumerate(observation_datetimes)
    }
    return panel_arr, target_arr, datetime_to_position


def build_supervised_windows(
    target_datetimes: Iterable[pd.Timestamp],
    panel_arr: np.ndarray,
    target_arr: np.ndarray,
    datetime_to_position: dict[pd.Timestamp, int],
    time_window: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    used_datetimes: list[pd.Timestamp] = []

    for dt in target_datetimes:
        end = datetime_to_position[dt]
        start = end - (time_window - 1)
        if start < 0:
            continue
        window = panel_arr[start : end + 1]
        target = target_arr[end]
        if np.isnan(window).any() or np.isnan(target).any():
            continue
        x_list.append(window)
        y_list.append(target)
        used_datetimes.append(dt)

    num_nodes = panel_arr.shape[1]
    num_features = panel_arr.shape[2]
    x = (
        np.stack(x_list)
        if x_list
        else np.empty((0, time_window, num_nodes, num_features), dtype=np.float32)
    )
    y = (
        np.stack(y_list)
        if y_list
        else np.empty((0, num_nodes), dtype=np.float32)
    )
    return x, y, used_datetimes


def chronological_train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("Jumlah sample X dan y harus sama.")
    n_train, n_validation = chronological_split_sizes(
        x.shape[0], validation_ratio
    )
    return (
        x[:n_train],
        y[:n_train],
        x[n_train:],
        y[n_train:],
        n_train,
        n_validation,
    )
