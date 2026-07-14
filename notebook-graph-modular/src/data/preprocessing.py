"""Datetime processing, observation filtering, cleaning, and causal fills."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def process_datetime_column(
    frame: pd.DataFrame,
    column: str = "datetime",
) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise")
    return result


def build_observation_datetimes(
    start: pd.Timestamp,
    end: pd.Timestamp,
    observation_hours: Iterable[int],
    frequency: str = "6h",
) -> pd.DatetimeIndex:
    datetimes = pd.date_range(start, end, freq=frequency)
    hours = set(observation_hours)
    return datetimes[datetimes.hour.isin(hours)]


def filter_observation_rows(
    frame: pd.DataFrame,
    observation_datetimes: pd.DatetimeIndex,
    datetime_column: str = "datetime",
) -> pd.DataFrame:
    return frame[frame[datetime_column].isin(observation_datetimes)].copy()


def build_full_station_index(
    observation_datetimes: pd.DatetimeIndex,
    node_order: list[str],
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [observation_datetimes, node_order],
        names=["datetime", "nama_pos"],
    )


def forward_fill_environment(
    environment_observations: pd.DataFrame,
    full_index: pd.MultiIndex,
    feature_columns: list[str],
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    """Reindex environment data and reproduce the notebook's causal fill policy."""
    environment_full = (
        environment_observations.set_index(["datetime", "nama_pos"])
        .reindex(full_index)
        .reset_index()
    )
    missing_before = int(environment_full[feature_columns].isna().sum().sum())

    environment_full = environment_full.sort_values(["nama_pos", "datetime"])
    environment_full[feature_columns] = (
        environment_full.groupby("nama_pos")[feature_columns].ffill()
    )

    train_mask = environment_full["datetime"] <= train_end
    fallback_means = environment_full.loc[train_mask, feature_columns].mean()
    environment_full[feature_columns] = environment_full[feature_columns].fillna(
        fallback_means
    )
    missing_after = int(environment_full[feature_columns].isna().sum().sum())
    return environment_full, fallback_means, missing_before, missing_after

