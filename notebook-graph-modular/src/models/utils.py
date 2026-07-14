"""Small model-construction utilities without training behavior."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from .temporal_gnn import TemporalGNN


def build_temporal_gnn(
    num_nodes: int,
    num_features: int,
    normalized_adjacency: np.ndarray,
    gcn_hidden: int = 64,
    gru_hidden: int = 64,
    dropout: float = 0.2,
    spatial_residual: bool = False,
) -> TemporalGNN:
    return TemporalGNN(
        num_nodes,
        num_features,
        normalized_adjacency,
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
        dropout=dropout,
        spatial_residual=spatial_residual,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model
