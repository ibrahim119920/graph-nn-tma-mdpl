"""Dense normalized adjacency used by the baseline GNN."""

from __future__ import annotations

import numpy as np


def build_normalized_adjacency(
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    num_nodes: int,
) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge_position in range(edge_index.shape[1]):
        source = edge_index[0, edge_position]
        target = edge_index[1, edge_position]
        weight = (
            edge_weight[edge_position]
            if edge_weight is not None
            else 1.0
        )
        weight = 1.0 / (weight + 1e-3)
        adjacency[source, target] = max(
            adjacency[source, target], weight
        )

    adjacency = adjacency + np.eye(num_nodes, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree_inverse_sqrt = np.power(degree, -0.5, where=degree > 0)
    degree_inverse_sqrt[degree == 0] = 0.0
    degree_matrix = np.diag(degree_inverse_sqrt)
    normalized = degree_matrix @ adjacency @ degree_matrix
    return normalized.astype(np.float32)

