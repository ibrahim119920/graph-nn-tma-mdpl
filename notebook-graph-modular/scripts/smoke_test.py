"""Fast import/build/forward smoke test without training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.graph.adjacency import build_normalized_adjacency
from src.models import TemporalGNN
from src.utils.config import load_project_config


def run_smoke_test() -> tuple[int, int]:
    config = load_project_config(PROJECT_ROOT / "configs" / "default.json")
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight = np.array([2.0, 2.0], dtype=np.float32)
    adjacency = build_normalized_adjacency(edge_index, edge_weight, 2)
    model = TemporalGNN(
        2,
        3,
        adjacency,
        gcn_hidden=config.model.gcn_hidden,
        gru_hidden=config.model.gru_hidden,
        dropout=config.model.dropout,
    ).eval()
    features = torch.zeros((1, 4, 2, 3), dtype=torch.float32)
    with torch.no_grad():
        output = model(features)
    if output.shape != (1, 2) or not torch.isfinite(output).all():
        raise RuntimeError(f"Smoke test gagal: output {output.shape}")
    return tuple(output.shape)


def main() -> None:
    shape = run_smoke_test()
    print(f"SMOKE_TEST_OK output_shape={shape}")


if __name__ == "__main__":
    main()

