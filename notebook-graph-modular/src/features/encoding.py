"""Baseline cyclic datetime encoding and spatial feature preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_time_features(
    observation_datetimes: pd.DatetimeIndex,
) -> pd.DataFrame:
    features = pd.DataFrame({"datetime": observation_datetimes})
    features["hour_sin"] = np.sin(
        2 * np.pi * features["datetime"].dt.hour / 24
    )
    features["hour_cos"] = np.cos(
        2 * np.pi * features["datetime"].dt.hour / 24
    )
    day_of_year = features["datetime"].dt.dayofyear
    days_in_year = np.where(features["datetime"].dt.is_leap_year, 366, 365)
    features["dayofyear_sin"] = np.sin(
        2 * np.pi * day_of_year / days_in_year
    )
    features["dayofyear_cos"] = np.cos(
        2 * np.pi * day_of_year / days_in_year
    )
    return features


def build_spatial_features(coordinate_index: pd.DataFrame) -> pd.DataFrame:
    return coordinate_index[["latitude", "longitude"]].reset_index()

