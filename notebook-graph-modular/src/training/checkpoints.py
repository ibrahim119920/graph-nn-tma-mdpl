"""Checkpoint creation, save/load, and baseline model restoration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.graph.adjacency import build_normalized_adjacency
from src.models import TemporalGNN
from src.models.utils import unwrap_model


def save_checkpoint(path: str | Path, payload: dict) -> None:
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    map_location: torch.device | str,
) -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_baseline_checkpoint_payload(
    model: nn.Module,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    station_center: np.ndarray,
    station_scale: np.ndarray,
    station_low: np.ndarray,
    station_high: np.ndarray,
    water_value_positions: list[int],
    water_std_positions: list[int],
    normalized_adjacency: np.ndarray,
    num_nodes: int,
    num_features: int,
    time_window: int,
    node_order: list[str] | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "feat_mean": feature_mean,
        "feat_std": feature_std,
        "station_center": station_center,
        "station_scale": station_scale,
        "station_low": station_low,
        "station_high": station_high,
        "wl_value_positions": np.array(water_value_positions),
        "wl_std_positions": np.array(water_std_positions),
        "A_norm": normalized_adjacency,
        "num_nodes": num_nodes,
        "num_features": num_features,
        "time_window": time_window,
    }
    if node_order is not None:
        payload["node_order"] = list(node_order)
    if feature_columns is not None:
        payload["feature_cols"] = list(feature_columns)
    return payload


def validate_checkpoint_dataset_alignment(
    checkpoint: dict,
    dataset: dict[str, np.ndarray],
) -> None:
    node_order = list(dataset["node_order"])
    feature_columns = list(dataset["feature_cols"])
    if len(node_order) != int(checkpoint["num_nodes"]):
        raise ValueError("Jumlah node checkpoint dan dataset berbeda.")
    if len(feature_columns) != int(checkpoint["num_features"]):
        raise ValueError("Jumlah fitur checkpoint dan dataset berbeda.")
    if checkpoint["A_norm"].shape != (len(node_order), len(node_order)):
        raise ValueError("Shape adjacency checkpoint tidak cocok dengan node order.")
    if "edge_index" in dataset:
        expected_adjacency = build_normalized_adjacency(
            dataset["edge_index"], dataset.get("edge_weight"), len(node_order)
        )
        if not np.allclose(
            checkpoint["A_norm"], expected_adjacency, rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Adjacency checkpoint dan graph dataset berbeda.")
    if "node_order" in checkpoint and list(checkpoint["node_order"]) != node_order:
        raise ValueError("Urutan node checkpoint dan dataset berbeda.")
    if "feature_cols" in checkpoint and list(checkpoint["feature_cols"]) != feature_columns:
        raise ValueError("Urutan fitur checkpoint dan dataset berbeda.")
    for key in ("wl_value_positions", "wl_std_positions"):
        positions = np.asarray(checkpoint[key])
        if positions.size and (
            positions.min() < 0 or positions.max() >= len(feature_columns)
        ):
            raise ValueError(f"{key} checkpoint berada di luar feature schema.")


def load_temporal_gnn_from_checkpoint(
    checkpoint: dict,
    device: torch.device | str,
    *,
    gcn_hidden: int = 64,
    gru_hidden: int = 64,
    dropout: float = 0.0,
) -> TemporalGNN:
    model = TemporalGNN(
        checkpoint["num_nodes"],
        checkpoint["num_features"],
        checkpoint["A_norm"],
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
