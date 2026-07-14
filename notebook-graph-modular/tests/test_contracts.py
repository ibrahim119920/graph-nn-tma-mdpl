import unittest
import warnings

import numpy as np
import torch

from src.data.loading import validate_graph_dataset
from src.graph.adjacency import build_normalized_adjacency
from src.models import TemporalGNN


def _dataset() -> dict[str, np.ndarray]:
    return {
        "X_train": np.zeros((2, 3, 2, 2), dtype=np.float32),
        "y_train": np.zeros((2, 2), dtype=np.float32),
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "edge_weight": np.ones(2, dtype=np.float32),
        "node_order": np.array(["A", "B"], dtype=object),
        "feature_cols": np.array(["wl_t", "rainfall"], dtype=object),
        "dt_train": np.array(["2025-01-01", "2025-01-02"]),
        "panel_arr": np.zeros((4, 2, 2), dtype=np.float32),
        "obs_datetimes": np.array(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        ),
    }


class DatasetAndModelContractTest(unittest.TestCase):
    def test_dataset_contract_accepts_aligned_arrays(self):
        validate_graph_dataset(_dataset())

    def test_dataset_contract_rejects_feature_width_mismatch(self):
        dataset = _dataset()
        dataset["feature_cols"] = np.array(["wl_t"], dtype=object)
        with self.assertRaisesRegex(ValueError, "feature_cols"):
            validate_graph_dataset(dataset)

    def test_adjacency_rejects_out_of_range_node(self):
        with self.assertRaisesRegex(ValueError, "luar rentang"):
            build_normalized_adjacency(
                np.array([[0], [2]], dtype=np.int64),
                np.array([1.0], dtype=np.float32),
                2,
            )

    def test_adjacency_normalization_has_no_uninitialized_where_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalized = build_normalized_adjacency(
                np.array([[0], [1]], dtype=np.int64),
                np.array([1.0], dtype=np.float32),
                2,
            )
        self.assertTrue(np.isfinite(normalized).all())
        self.assertFalse(
            any("where" in str(warning.message) for warning in caught)
        )

    def test_model_contract_rejects_wrong_feature_count(self):
        adjacency = np.eye(2, dtype=np.float32)
        model = TemporalGNN(2, 3, adjacency).eval()
        with self.assertRaisesRegex(ValueError, "nodes/features"):
            model(torch.zeros((1, 2, 2, 4), dtype=torch.float32))

    def test_spatial_residual_model_preserves_output_contract(self):
        model = TemporalGNN(
            2,
            3,
            np.eye(2, dtype=np.float32),
            spatial_residual=True,
        ).eval()
        with torch.no_grad():
            output = model(torch.zeros((1, 2, 2, 3), dtype=torch.float32))
        self.assertTrue(model.spatial_residual)
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
