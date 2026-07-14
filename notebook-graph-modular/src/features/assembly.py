"""Assembly of long-format node features and next-step target."""

from __future__ import annotations

import pandas as pd


def assemble_node_features(
    water_features: pd.DataFrame,
    environment_features: pd.DataFrame,
    time_features: pd.DataFrame,
    spatial_features: pd.DataFrame,
    river_attributes: pd.DataFrame,
) -> pd.DataFrame:
    result = water_features.merge(
        environment_features, on=["datetime", "nama_pos"], how="left"
    )
    result = result.merge(time_features, on="datetime", how="left")
    result = result.merge(spatial_features, on="nama_pos", how="left")
    result = result.merge(
        river_attributes.reset_index(), on="nama_pos", how="left"
    )
    result = result.sort_values(["nama_pos", "datetime"])
    result["y_next"] = result.groupby("nama_pos")["wl_t"].shift(-1)
    return result


def build_feature_columns(
    lag_columns: list[str],
    rolling_columns: list[str],
    environment_columns: list[str],
    river_columns: list[str],
) -> list[str]:
    return (
        ["wl_t"]
        + lag_columns
        + rolling_columns
        + environment_columns
        + ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]
        + ["latitude", "longitude"]
        + river_columns
    )

