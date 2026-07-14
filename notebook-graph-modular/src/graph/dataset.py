"""PyTorch graph time-series dataset and DataLoader construction."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GraphTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(targets)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def build_graph_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    batch_size: int,
) -> tuple[
    GraphTimeSeriesDataset,
    GraphTimeSeriesDataset,
    DataLoader,
    DataLoader,
]:
    train_dataset = GraphTimeSeriesDataset(x_train, y_train)
    validation_dataset = GraphTimeSeriesDataset(x_validation, y_validation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )

