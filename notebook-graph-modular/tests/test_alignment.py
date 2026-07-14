import unittest

import numpy as np

from src.training.checkpoints import validate_checkpoint_dataset_alignment


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = {
            "num_nodes": 2,
            "num_features": 3,
            "A_norm": np.eye(2, dtype=np.float32),
            "node_order": ["A", "B"],
            "feature_cols": ["wl_t", "x", "y"],
            "wl_value_positions": np.array([0]),
            "wl_std_positions": np.array([], dtype=np.int64),
        }
        self.dataset = {
            "node_order": np.array(["A", "B"], dtype=object),
            "feature_cols": np.array(["wl_t", "x", "y"], dtype=object),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_weight": np.empty(0, dtype=np.float32),
        }

    def test_matching_alignment_passes(self):
        validate_checkpoint_dataset_alignment(self.checkpoint, self.dataset)

    def test_permuted_nodes_fail(self):
        self.dataset["node_order"] = np.array(["B", "A"], dtype=object)
        with self.assertRaisesRegex(ValueError, "Urutan node"):
            validate_checkpoint_dataset_alignment(self.checkpoint, self.dataset)

    def test_different_graph_fails(self):
        self.dataset["edge_index"] = np.array([[0], [1]], dtype=np.int64)
        self.dataset["edge_weight"] = np.array([1.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Adjacency"):
            validate_checkpoint_dataset_alignment(self.checkpoint, self.dataset)


if __name__ == "__main__":
    unittest.main()
