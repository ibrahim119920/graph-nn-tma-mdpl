"""Baseline Temporal GNN extracted from the notebook."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .layers import GCNLayer


class TemporalGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        normalized_adjacency: np.ndarray,
        gcn_hidden: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.register_buffer(
            "A_norm", torch.from_numpy(normalized_adjacency)
        )
        self.num_nodes = num_nodes

        self.gcn1 = GCNLayer(num_features, gcn_hidden)
        self.local_skip = nn.Linear(num_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, _num_features = features.shape
        gcn_output: list[torch.Tensor] = []
        for time_index in range(time_steps):
            hidden = self.gcn1(
                features[:, time_index], self.A_norm
            ) + self.local_skip(features[:, time_index])
            hidden = self.act(hidden)
            hidden = self.drop(hidden)
            hidden = self.gcn2(hidden, self.A_norm)
            hidden = self.act(hidden)
            gcn_output.append(hidden)

        hidden_sequence = torch.stack(gcn_output, dim=1)
        hidden_sequence = hidden_sequence.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes,
            time_steps,
            -1,
        )
        _, hidden_n = self.gru(hidden_sequence)
        hidden_last = hidden_n.squeeze(0)
        output = self.head(hidden_last).view(batch_size, num_nodes)
        return output

