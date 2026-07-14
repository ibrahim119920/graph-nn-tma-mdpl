"""Dense normalized adjacency used by the baseline GNN."""

from __future__ import annotations

import numpy as np


def build_normalized_adjacency(
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    num_nodes: int,
) -> np.ndarray:
    if num_nodes <= 0:
        raise ValueError(f"num_nodes harus positif, diterima: {num_nodes}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index harus berbentuk (2,E), diterima: {edge_index.shape}")
    if edge_weight is not None and len(edge_weight) != edge_index.shape[1]:
        raise ValueError(
            "Panjang edge_weight harus sama dengan jumlah edge: "
            f"{len(edge_weight)} != {edge_index.shape[1]}"
        )
    if edge_index.size and (
        edge_index.min() < 0 or edge_index.max() >= num_nodes
    ):
        raise ValueError("edge_index berisi indeks node di luar rentang.")
    if edge_weight is not None and (
        not np.isfinite(edge_weight).all() or (edge_weight < 0).any()
    ):
        raise ValueError("edge_weight harus finite dan non-negatif.")
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
    degree_inverse_sqrt = np.zeros_like(degree)
    np.power(
        degree,
        -0.5,
        out=degree_inverse_sqrt,
        where=degree > 0,
    )
    degree_matrix = np.diag(degree_inverse_sqrt)
    normalized = degree_matrix @ adjacency @ degree_matrix
    return normalized.astype(np.float32)
