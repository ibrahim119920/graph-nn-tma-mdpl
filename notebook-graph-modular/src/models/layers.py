"""Graph neural-network layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.einsum("ij,bjf->bif", normalized_adjacency, x)
        return self.linear(x)

