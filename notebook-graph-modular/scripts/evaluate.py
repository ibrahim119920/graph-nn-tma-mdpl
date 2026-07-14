"""Evaluate a baseline checkpoint on the chronological validation tail."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.loading import load_numpy_dataset
from src.data.splitting import chronological_train_validation_split
from src.evaluation import evaluate_loader
from src.graph.dataset import build_graph_dataloaders
from src.training import (
    load_checkpoint,
    load_temporal_gnn_from_checkpoint,
    validate_checkpoint_dataset_alignment,
)
from src.utils.config import ProjectConfig
from src.utils.runtime import select_device
from src.utils.scaling import GraphFeatureScaler


def evaluate_checkpoint(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config: ProjectConfig | None = None,
    logger=None,
):
    device = torch.device(device) if device is not None else select_device()
    data = load_numpy_dataset(dataset_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_checkpoint_dataset_alignment(checkpoint, data)
    scaler = GraphFeatureScaler.from_checkpoint(checkpoint)
    features = scaler.normalize_features(data["X_train"]).astype(np.float32)
    targets = scaler.normalize_target(data["y_train"]).astype(np.float32)
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        _,
        _,
    ) = chronological_train_validation_split(
        features,
        targets,
        config.training.validation_ratio if config else 0.1,
    )
    _, _, _, validation_loader = build_graph_dataloaders(
        x_train,
        y_train,
        x_validation,
        y_validation,
        batch_size=config.training.batch_size if config else 64,
    )
    model = load_temporal_gnn_from_checkpoint(
        checkpoint,
        device,
        gcn_hidden=config.model.gcn_hidden if config else 64,
        gru_hidden=config.model.gru_hidden if config else 64,
        dropout=0.0,
    )
    result = evaluate_loader(
        model, validation_loader, device, scaler.denormalize_target
    )
    emit = logger.info if logger else print
    emit(f"RMSE: {result.rmse:.4f}")
    emit(f"MAE : {result.mae:.4f}")
    for index, name in enumerate(list(data["node_order"])):
        emit(f"  {name:35s} RMSE={result.per_node_rmse[index]:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.json"),
    )
    args = parser.parse_args()
    from scripts.run_pipeline import run_from_config

    run_from_config(args.config, stage="evaluate")


if __name__ == "__main__":
    main()
