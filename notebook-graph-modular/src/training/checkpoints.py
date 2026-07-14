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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    map_location: torch.device | str,
) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint model tidak ditemukan: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except Exception as error:
        raise ValueError(
            f"Checkpoint tidak dapat dimuat: {checkpoint_path} ({error})"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint harus berupa dictionary: {checkpoint_path}")
    validate_checkpoint_payload(checkpoint, checkpoint_path)
    return checkpoint


REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "model_state_dict",
        "feat_mean",
        "feat_std",
        "station_center",
        "station_scale",
        "station_low",
        "station_high",
        "wl_value_positions",
        "wl_std_positions",
        "A_norm",
        "num_nodes",
        "num_features",
        "time_window",
    }
)


def validate_checkpoint_payload(
    checkpoint: dict,
    checkpoint_path: str | Path = "<in-memory>",
) -> None:
    """Fail early when an old/corrupt checkpoint cannot satisfy this model."""
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise ValueError(
            f"Checkpoint tidak kompatibel ({checkpoint_path}); key hilang: {missing}"
        )
    num_nodes = int(checkpoint["num_nodes"])
    num_features = int(checkpoint["num_features"])
    time_window = int(checkpoint["time_window"])
    if num_nodes <= 0 or num_features <= 0 or time_window <= 0:
        raise ValueError(
            f"Checkpoint tidak kompatibel ({checkpoint_path}); "
            f"num_nodes/num_features/time_window harus positif, got "
            f"{num_nodes}/{num_features}/{time_window}."
        )
    adjacency = np.asarray(checkpoint["A_norm"])
    if adjacency.shape != (num_nodes, num_nodes) or not np.isfinite(adjacency).all():
        raise ValueError(
            f"Checkpoint tidak kompatibel ({checkpoint_path}); A_norm harus finite "
            f"dengan shape ({num_nodes}, {num_nodes}), got {adjacency.shape}."
        )
    for key in ("station_center", "station_scale", "station_low", "station_high"):
        values = np.asarray(checkpoint[key])
        if values.shape != (num_nodes,) or not np.isfinite(values).all():
            raise ValueError(
                f"Checkpoint tidak kompatibel ({checkpoint_path}); {key} harus "
                f"finite dengan shape ({num_nodes},), got {values.shape}."
            )
    if (np.asarray(checkpoint["station_scale"]) <= 0).any():
        raise ValueError(
            f"Checkpoint tidak kompatibel ({checkpoint_path}); station_scale harus positif."
        )
    for key in ("feat_mean", "feat_std"):
        values = np.asarray(checkpoint[key])
        if values.shape != (num_features,) or not np.isfinite(values).all():
            raise ValueError(
                f"Checkpoint tidak kompatibel ({checkpoint_path}); {key} harus "
                f"finite dengan shape ({num_features},), got {values.shape}."
            )
    if (np.asarray(checkpoint["feat_std"]) <= 0).any():
        raise ValueError(
            f"Checkpoint tidak kompatibel ({checkpoint_path}); feat_std harus positif."
        )
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError(
            f"Checkpoint tidak kompatibel ({checkpoint_path}); model_state_dict bukan dictionary."
        )


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
    spatial_residual: bool = False,
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
        "spatial_residual": spatial_residual,
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
    validate_checkpoint_payload(checkpoint)
    node_order = list(dataset["node_order"])
    feature_columns = list(dataset["feature_cols"])
    if len(node_order) != int(checkpoint["num_nodes"]):
        raise ValueError("Jumlah node checkpoint dan dataset berbeda.")
    if len(feature_columns) != int(checkpoint["num_features"]):
        raise ValueError("Jumlah fitur checkpoint dan dataset berbeda.")
    if "X_train" in dataset and np.asarray(dataset["X_train"]).shape[1] != int(
        checkpoint["time_window"]
    ):
        raise ValueError(
            "time_window checkpoint dan dataset berbeda: "
            f"{checkpoint['time_window']} != "
            f"{np.asarray(dataset['X_train']).shape[1]}"
        )
    checkpoint_adjacency = np.asarray(checkpoint["A_norm"])
    if checkpoint_adjacency.shape != (len(node_order), len(node_order)):
        raise ValueError("Shape adjacency checkpoint tidak cocok dengan node order.")
    if "edge_index" in dataset:
        expected_adjacency = build_normalized_adjacency(
            dataset["edge_index"], dataset.get("edge_weight"), len(node_order)
        )
        if not np.allclose(
            checkpoint_adjacency, expected_adjacency, rtol=1e-6, atol=1e-7
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


def validate_checkpoint_model_compatibility(
    checkpoint: dict,
    *,
    gcn_hidden: int,
    gru_hidden: int,
    spatial_residual: bool | None = None,
) -> None:
    """Check configured hidden dimensions against serialized state before load."""
    validate_checkpoint_payload(checkpoint)
    state_dict = checkpoint["model_state_dict"]
    expected_shapes = {
        "gcn1.linear.weight": (gcn_hidden, int(checkpoint["num_features"])),
        "local_skip.weight": (gcn_hidden, int(checkpoint["num_features"])),
        "gcn2.linear.weight": (gcn_hidden, gcn_hidden),
        "gru.weight_ih_l0": (3 * gru_hidden, gcn_hidden),
    }
    for key, expected_shape in expected_shapes.items():
        value = state_dict.get(key)
        actual_shape = tuple(value.shape) if value is not None else None
        if actual_shape != expected_shape:
            raise ValueError(
                "Checkpoint tidak kompatibel dengan model config: "
                f"{key} memiliki shape {actual_shape}, mengharapkan {expected_shape}."
            )
    if (
        spatial_residual is not None
        and bool(checkpoint.get("spatial_residual", False))
        != spatial_residual
    ):
        raise ValueError(
            "Checkpoint tidak kompatibel dengan model config: "
            "spatial_residual berbeda."
        )


def load_temporal_gnn_from_checkpoint(
    checkpoint: dict,
    device: torch.device | str,
    *,
    gcn_hidden: int = 64,
    gru_hidden: int = 64,
    dropout: float = 0.0,
) -> TemporalGNN:
    validate_checkpoint_model_compatibility(
        checkpoint, gcn_hidden=gcn_hidden, gru_hidden=gru_hidden
    )
    model = TemporalGNN(
        checkpoint["num_nodes"],
        checkpoint["num_features"],
        checkpoint["A_norm"],
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
        dropout=dropout,
        spatial_residual=bool(checkpoint.get("spatial_residual", False)),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint tidak kompatibel dengan arsitektur TemporalGNN: {error}"
        ) from error
    return model.to(device).eval()
