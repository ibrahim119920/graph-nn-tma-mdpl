"""Causal water-level lag and rolling feature engineering."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def build_water_level_features(
    train_observations: pd.DataFrame,
    full_index: pd.MultiIndex,
    n_lags: int,
    rolling_windows: Iterable[int],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    water = train_observations[["datetime", "nama_pos", "tma_mdpl"]].copy()
    water = (
        water.set_index(["datetime", "nama_pos"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["nama_pos", "datetime"])
        .rename(columns={"tma_mdpl": "wl_t"})
    )

    grouped = water.groupby("nama_pos")["wl_t"]
    for lag in range(1, n_lags + 1):
        water[f"wl_t-{lag}"] = grouped.shift(lag)

    water["_wl_shift1"] = grouped.shift(1)
    shifted_grouped = water.groupby("nama_pos")["_wl_shift1"]
    for window in rolling_windows:
        rolling = lambda series: series.rolling(window, min_periods=window)
        water[f"wl_roll{window}_mean"] = shifted_grouped.transform(
            lambda series: rolling(series).mean()
        )
        water[f"wl_roll{window}_std"] = shifted_grouped.transform(
            lambda series: rolling(series).std()
        )
        water[f"wl_roll{window}_min"] = shifted_grouped.transform(
            lambda series: rolling(series).min()
        )
        water[f"wl_roll{window}_max"] = shifted_grouped.transform(
            lambda series: rolling(series).max()
        )
    water = water.drop(columns=["_wl_shift1"])

    lag_columns = [f"wl_t-{lag}" for lag in range(1, n_lags + 1)]
    rolling_columns = [c for c in water.columns if c.startswith("wl_roll")]
    return water, lag_columns, rolling_columns

