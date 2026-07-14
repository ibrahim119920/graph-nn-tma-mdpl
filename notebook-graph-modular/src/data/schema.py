"""Schema and canonical station-order validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StationCoverage:
    missing_in_train: frozenset[str]
    missing_in_environment: frozenset[str]
    extra_in_train: frozenset[str]
    extra_in_environment: frozenset[str]


def validate_station_coverage(
    coordinate: pd.DataFrame,
    train_data: pd.DataFrame,
    environment_data: pd.DataFrame,
) -> StationCoverage:
    pos_coord = set(coordinate["nama_pos"].unique())
    pos_train = set(train_data["nama_pos"].unique())
    pos_env = set(environment_data["nama_pos"].unique())
    return StationCoverage(
        missing_in_train=frozenset(pos_coord - pos_train),
        missing_in_environment=frozenset(pos_coord - pos_env),
        extra_in_train=frozenset(pos_train - pos_coord),
        extra_in_environment=frozenset(pos_env - pos_coord),
    )


def build_node_index(
    coordinate: pd.DataFrame,
    train_data: pd.DataFrame,
    environment_data: pd.DataFrame,
) -> tuple[list[str], dict[str, int], pd.DataFrame, StationCoverage]:
    """Create the alphabetical node order used by features, graph, and targets."""
    if coordinate["nama_pos"].duplicated().any():
        raise ValueError("nama_pos pada koordinat harus unik.")
    node_order = sorted(coordinate["nama_pos"].unique().tolist())
    node_to_idx = {name: i for i, name in enumerate(node_order)}
    coordinate_idx = coordinate.set_index("nama_pos").loc[node_order]
    coverage = validate_station_coverage(coordinate, train_data, environment_data)
    return node_order, node_to_idx, coordinate_idx, coverage

