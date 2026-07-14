"""Train the baseline Temporal GNN with the notebook methodology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.loading import load_numpy_dataset
from src.data.splitting import (
    chronological_split_sizes,
    chronological_train_validation_split,
)
from src.evaluation import RolloutValidationContext, evaluate_rollout
from src.graph.adjacency import build_normalized_adjacency
from src.graph.dataset import build_graph_dataloaders
from src.inference.rollout import AutoregressiveFeatureSpec
from src.models import build_temporal_gnn
from src.training import (
    BestCheckpointCallback,
    EarlyStopping,
    build_baseline_checkpoint_payload,
    fit_with_rollout,
)
from src.utils.config import ProjectConfig
from src.utils.runtime import select_device
from src.utils.scaling import GraphFeatureScaler


BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15
VALIDATION_RATIO = 0.1
N_LAGS = 11


def train_baseline(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config: ProjectConfig | None = None,
    logger=None,
) -> dict:
    device = torch.device(device) if device is not None else select_device()
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    batch_size = config.training.batch_size if config else BATCH_SIZE
    epochs = config.training.epochs if config else EPOCHS
    learning_rate = (
        config.training.learning_rate if config else LEARNING_RATE
    )
    weight_decay = config.training.weight_decay if config else 1e-5
    patience = config.training.patience if config else PATIENCE
    validation_ratio = (
        config.training.validation_ratio if config else VALIDATION_RATIO
    )
    huber_beta = config.training.huber_beta if config else 0.5
    scheduler_factor = (
        config.training.scheduler_factor if config else 0.5
    )
    scheduler_patience = (
        config.training.scheduler_patience if config else 5
    )
    n_lags = config.inference.n_lags if config else N_LAGS
    gcn_hidden = config.model.gcn_hidden if config else 64
    gru_hidden = config.model.gru_hidden if config else 64
    dropout = config.model.dropout if config else 0.2

    def emit(message: str) -> None:
        logger.info(message) if logger else print(message)

    data = load_numpy_dataset(dataset_path)
    features = data["X_train"]
    targets = data["y_train"]
    feature_columns = list(data["feature_cols"])
    node_order = list(data["node_order"])
    training_datetimes = pd.to_datetime(data["dt_train"])
    panel = data["panel_arr"].copy()
    observation_datetimes = pd.to_datetime(data["obs_datetimes"])

    n_train, n_validation = chronological_split_sizes(
        features.shape[0], validation_ratio
    )
    scaler = GraphFeatureScaler.fit(features[:n_train], feature_columns)
    normalized_features = scaler.normalize_features(features).astype(np.float32)
    normalized_targets = scaler.normalize_target(targets).astype(np.float32)
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        _,
        _,
    ) = chronological_train_validation_split(
        normalized_features,
        normalized_targets,
        validation_ratio,
    )
    _, _, train_loader, validation_loader = build_graph_dataloaders(
        x_train,
        y_train,
        x_validation,
        y_validation,
        batch_size,
        seed=config.runtime.seed if config else None,
    )

    normalized_adjacency = build_normalized_adjacency(
        data["edge_index"], data["edge_weight"], features.shape[2]
    )
    model = build_temporal_gnn(
        features.shape[2],
        features.shape[3],
        normalized_adjacency,
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
        dropout=dropout,
    )
    if gpu_count >= 2:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    model = model.to(device)

    criterion = nn.SmoothL1Loss(beta=huber_beta)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    rollout_context = RolloutValidationContext(
        panel=panel,
        observation_datetimes=observation_datetimes,
        training_datetimes=training_datetimes,
        validation_start_index=n_train,
        time_window=features.shape[1],
        feature_spec=AutoregressiveFeatureSpec.from_feature_columns(
            feature_columns, n_lags
        ),
        station_center=scaler.station_center,
        station_scale=scaler.station_scale,
        station_low=scaler.station_low,
        station_high=scaler.station_high,
        normalize_features=scaler.normalize_features,
    )

    checkpoint_callback = BestCheckpointCallback(
        checkpoint_path,
        lambda current_model: build_baseline_checkpoint_payload(
            current_model,
            feature_mean=scaler.feature_mean,
            feature_std=scaler.feature_std,
            station_center=scaler.station_center,
            station_scale=scaler.station_scale,
            station_low=scaler.station_low,
            station_high=scaler.station_high,
            water_value_positions=scaler.water_value_positions,
            water_std_positions=scaler.water_std_positions,
            normalized_adjacency=normalized_adjacency,
            num_nodes=features.shape[2],
            num_features=features.shape[3],
            time_window=features.shape[1],
            node_order=node_order,
            feature_columns=feature_columns,
        ),
    )

    def print_epoch(record) -> None:
        emit(
            f"Epoch {record.epoch:3d}/{epochs} | "
            f"train_loss={record.train_loss:.5f} | "
            f"teacher_val={record.teacher_validation_loss:.5f} | "
            f"rollout_val={record.rollout_validation_loss:.5f} | "
            f"lr={record.learning_rate:.2e} | "
            f"{record.duration_seconds:.1f}s"
            + ("  <- best" if record.improved else "")
        )

    emit(f"Device: {device} | Jumlah GPU terdeteksi: {gpu_count}")
    emit(f"Train: {n_train} sample | Val: {n_validation} sample")
    result = fit_with_rollout(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=epochs,
        early_stopping=EarlyStopping(patience),
        rollout_metric=lambda: evaluate_rollout(
            model, rollout_context, device
        ),
        improvement_callback=checkpoint_callback,
        epoch_callback=print_epoch,
    )
    emit(
        f"Best rollout_val (normalized RMSE): {result.best_metric:.5f} | "
        f"checkpoint: {checkpoint_path}"
    )
    return {
        "result": result,
        "node_order": node_order,
        "checkpoint_path": Path(checkpoint_path),
        "checkpoint_metadata": {
            "path": Path(checkpoint_path),
            "best_epoch": result.best_epoch,
            "best_rollout_validation_rmse_normalized": result.best_metric,
            "num_nodes": features.shape[2],
            "num_features": features.shape[3],
            "time_window": features.shape[1],
            "node_order": node_order,
            "feature_columns": feature_columns,
            "split_summary": {
                "validation_ratio": validation_ratio,
                "train_samples": n_train,
                "validation_samples": n_validation,
                "train_datetime_start": str(training_datetimes[:n_train].min()),
                "train_datetime_end": str(training_datetimes[:n_train].max()),
                "validation_datetime_start": str(
                    training_datetimes[n_train:].min()
                ),
                "validation_datetime_end": str(
                    training_datetimes[n_train:].max()
                ),
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from scripts.run_pipeline import run_from_config

    run_from_config(args.config, stage="train")


if __name__ == "__main__":
    main()
