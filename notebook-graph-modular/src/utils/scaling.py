"""Per-station water-level and global feature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphFeatureScaler:
    station_center: np.ndarray
    station_scale: np.ndarray
    station_low: np.ndarray
    station_high: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    water_value_positions: list[int]
    water_std_positions: list[int]
    num_nodes: int

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "GraphFeatureScaler":
        return cls(
            station_center=checkpoint["station_center"],
            station_scale=checkpoint["station_scale"],
            station_low=checkpoint["station_low"],
            station_high=checkpoint["station_high"],
            feature_mean=checkpoint["feat_mean"],
            feature_std=checkpoint["feat_std"],
            water_value_positions=list(checkpoint["wl_value_positions"]),
            water_std_positions=list(checkpoint["wl_std_positions"]),
            num_nodes=int(checkpoint["num_nodes"]),
        )

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        feature_columns: list[str],
    ) -> "GraphFeatureScaler":
        if x_train.ndim != 4 or x_train.shape[0] == 0:
            raise ValueError("x_train harus berbentuk (N, T, nodes, features) dan tidak kosong.")
        num_nodes = x_train.shape[2]
        wl_t_position = feature_columns.index("wl_t")
        water_value_positions = [
            i
            for i, column in enumerate(feature_columns)
            if column == "wl_t"
            or column.startswith("wl_t-")
            or (column.startswith("wl_roll") and not column.endswith("_std"))
        ]
        water_std_positions = [
            i
            for i, column in enumerate(feature_columns)
            if column.startswith("wl_roll") and column.endswith("_std")
        ]

        water_values = x_train[:, :, :, wl_t_position].reshape(-1, num_nodes)
        station_center = np.nanmedian(water_values, axis=0).astype(np.float32)
        q25 = np.nanquantile(water_values, 0.25, axis=0)
        q75 = np.nanquantile(water_values, 0.75, axis=0)
        station_scale = (q75 - q25).astype(np.float32)
        station_std = np.nanstd(water_values, axis=0).astype(np.float32)
        station_scale = np.where(
            station_scale > 1e-3, station_scale, station_std
        )
        station_scale = np.where(
            station_scale > 1e-3, station_scale, 1.0
        ).astype(np.float32)

        q01 = np.nanquantile(water_values, 0.01, axis=0).astype(np.float32)
        q99 = np.nanquantile(water_values, 0.99, axis=0).astype(np.float32)
        station_low = q01 - 0.50 * station_scale
        station_high = q99 + 0.50 * station_scale

        provisional = cls(
            station_center=station_center,
            station_scale=station_scale,
            station_low=station_low,
            station_high=station_high,
            feature_mean=np.empty(x_train.shape[3], dtype=np.float32),
            feature_std=np.empty(x_train.shape[3], dtype=np.float32),
            water_value_positions=water_value_positions,
            water_std_positions=water_std_positions,
            num_nodes=num_nodes,
        )
        station_normalized = provisional.station_normalize_water_features(x_train)
        flattened = station_normalized.reshape(-1, x_train.shape[3])
        provisional.feature_mean = flattened.mean(axis=0).astype(np.float32)
        provisional.feature_std = flattened.std(axis=0).astype(np.float32)
        provisional.feature_std[provisional.feature_std < 1e-6] = 1.0
        return provisional

    def station_normalize_water_features(self, x: np.ndarray) -> np.ndarray:
        result = x.copy()
        center = self.station_center.reshape(1, 1, self.num_nodes)
        scale = self.station_scale.reshape(1, 1, self.num_nodes)
        for position in self.water_value_positions:
            result[..., position] = (result[..., position] - center) / scale
        for position in self.water_std_positions:
            result[..., position] = result[..., position] / scale
        return result

    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        normalized = self.station_normalize_water_features(x)
        return (normalized - self.feature_mean) / self.feature_std

    def normalize_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.station_center.reshape(1, -1)) / self.station_scale.reshape(1, -1)

    def denormalize_target(self, y: np.ndarray) -> np.ndarray:
        return y * self.station_scale.reshape(1, -1) + self.station_center.reshape(1, -1)
