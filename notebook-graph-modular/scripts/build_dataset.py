"""Build the graph dataset artifact with the original notebook methodology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.loading import (
    load_environment_data,
    load_hydrorivers,
    load_station_coordinates,
    load_train_data,
)
from src.data.preprocessing import (
    build_full_station_index,
    build_observation_datetimes,
    filter_observation_rows,
    forward_fill_environment,
    process_datetime_column,
)
from src.data.schema import build_node_index
from src.data.splitting import (
    build_feature_panel,
    build_supervised_windows,
    find_complete_timesteps,
)
from src.features.assembly import assemble_node_features, build_feature_columns
from src.features.encoding import build_spatial_features, build_time_features
from src.features.environment import DEFAULT_ENVIRONMENT_FEATURES
from src.features.water_level import build_water_level_features
from src.graph.construction import build_river_graph
from src.utils.config import ProjectConfig


OBSERVATION_HOURS = [6, 12, 18]
TRAIN_START = pd.Timestamp("2023-01-01 06:00:00")
TRAIN_END = pd.Timestamp("2025-09-18 18:00:00")
TEST_START = pd.Timestamp("2025-09-19 06:00:00")
TEST_END = pd.Timestamp("2026-05-18 18:00:00")
ROLLING_WINDOWS = [4, 8, 28]
MAX_DOWNSTREAM_HOPS = 3000


def build_dataset_artifact(
    raw_data_root: str | Path,
    output_path: str | Path,
    config: ProjectConfig,
    logger=None,
) -> dict:
    emit = logger.info if logger else print
    root = Path(raw_data_root)
    hydrorivers = load_hydrorivers(root)
    coordinates = load_station_coordinates(root)
    train_data = process_datetime_column(load_train_data(root))
    environment_data = process_datetime_column(
        load_environment_data(root, DEFAULT_ENVIRONMENT_FEATURES)
    )
    node_order, node_to_index, coordinate_index, coverage = build_node_index(
        coordinates, train_data, environment_data
    )
    if coverage.missing_in_train or coverage.missing_in_environment:
        raise ValueError(f"Cakupan stasiun tidak lengkap: {coverage}")

    observation_datetimes = build_observation_datetimes(
        TRAIN_START, TEST_END, OBSERVATION_HOURS
    )
    train_observations = filter_observation_rows(
        train_data, observation_datetimes
    )
    environment_observations = filter_observation_rows(
        environment_data, observation_datetimes
    )
    full_index = build_full_station_index(
        observation_datetimes, node_order
    )
    environment_full, _, _, missing_after = forward_fill_environment(
        environment_observations,
        full_index,
        DEFAULT_ENVIRONMENT_FEATURES.copy(),
        TRAIN_END,
    )
    if missing_after:
        raise ValueError(
            f"Fitur lingkungan masih memiliki {missing_after} nilai kosong."
        )

    graph = build_river_graph(
        hydrorivers,
        coordinate_index,
        node_order,
        node_to_index,
        MAX_DOWNSTREAM_HOPS,
    )
    water_features, lag_columns, rolling_columns = (
        build_water_level_features(
            train_observations,
            full_index,
            config.inference.n_lags,
            ROLLING_WINDOWS,
        )
    )
    time_features = build_time_features(observation_datetimes)
    spatial_features = build_spatial_features(coordinate_index)
    node_features = assemble_node_features(
        water_features,
        environment_full,
        time_features,
        spatial_features,
        graph.river_attributes,
    )
    feature_columns = build_feature_columns(
        lag_columns,
        rolling_columns,
        DEFAULT_ENVIRONMENT_FEATURES.copy(),
        list(graph.river_attributes.columns),
    )
    water_columns = ["wl_t"] + lag_columns + rolling_columns
    non_water_columns = [
        column for column in feature_columns if column not in water_columns
    ]
    train_valid_timesteps = find_complete_timesteps(
        node_features,
        TRAIN_START,
        TRAIN_END,
        feature_columns + ["y_next"],
    )
    test_valid_timesteps = find_complete_timesteps(
        node_features,
        TEST_START,
        TEST_END,
        non_water_columns,
    )
    panel, target, datetime_to_position = build_feature_panel(
        node_features,
        observation_datetimes,
        node_order,
        feature_columns,
    )
    features, targets, training_datetimes = build_supervised_windows(
        train_valid_timesteps,
        panel,
        target,
        datetime_to_position,
        config.inference.time_window,
    )
    water_column_positions = [
        feature_columns.index(column) for column in water_columns
    ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        X_train=features,
        y_train=targets,
        dt_train=np.array(pd.DatetimeIndex(training_datetimes).astype(str)),
        panel_arr=panel,
        obs_datetimes=np.array(observation_datetimes.astype(str)),
        test_valid_timesteps=np.array(test_valid_timesteps.astype(str)),
        wl_col_positions=np.array(water_column_positions),
        feature_cols=np.array(feature_columns, dtype=object),
        node_order=np.array(node_order, dtype=object),
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )
    emit(
        f"Dataset tersimpan: {destination} | X={features.shape} "
        f"y={targets.shape} edges={graph.edge_index.shape}"
    )
    return {
        "path": destination,
        "x_shape": features.shape,
        "y_shape": targets.shape,
        "edge_shape": graph.edge_index.shape,
        "node_count": len(node_order),
        "feature_count": len(feature_columns),
        "train_timesteps": len(training_datetimes),
        "test_timesteps": len(test_valid_timesteps),
        "node_order": node_order,
        "feature_columns": feature_columns,
        "observation_datetime_start": str(observation_datetimes.min()),
        "observation_datetime_end": str(observation_datetimes.max()),
        "training_datetime_start": str(pd.DatetimeIndex(training_datetimes).min()),
        "training_datetime_end": str(pd.DatetimeIndex(training_datetimes).max()),
        "test_datetime_start": str(test_valid_timesteps.min()),
        "test_datetime_end": str(test_valid_timesteps.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.json"),
    )
    args = parser.parse_args()
    from scripts.run_pipeline import run_from_config

    run_from_config(args.config, stage="build")


if __name__ == "__main__":
    main()
