import unittest

import numpy as np
import torch

from src.models import TemporalGNN
from src.training.checkpoints import (
    load_temporal_gnn_from_checkpoint,
    validate_checkpoint_dataset_alignment,
)


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = {
            "model_state_dict": {},
            "num_nodes": 2,
            "num_features": 3,
            "A_norm": np.eye(2, dtype=np.float32),
            "feat_mean": np.zeros(3, dtype=np.float32),
            "feat_std": np.ones(3, dtype=np.float32),
            "station_center": np.zeros(2, dtype=np.float32),
            "station_scale": np.ones(2, dtype=np.float32),
            "station_low": np.full(2, -1.0, dtype=np.float32),
            "station_high": np.full(2, 1.0, dtype=np.float32),
            "time_window": 2,
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

    def test_spatial_residual_checkpoint_mismatch_fails(self):
        from src.training.checkpoints import validate_checkpoint_model_compatibility

        self.checkpoint["model_state_dict"] = {
            "gcn1.linear.weight": np.zeros((4, 3), dtype=np.float32),
            "local_skip.weight": np.zeros((4, 3), dtype=np.float32),
            "gcn2.linear.weight": np.zeros((4, 4), dtype=np.float32),
            "gru.weight_ih_l0": np.zeros((15, 4), dtype=np.float32),
        }
        self.checkpoint["spatial_residual"] = False
        with self.assertRaisesRegex(ValueError, "spatial_residual"):
            validate_checkpoint_model_compatibility(
                self.checkpoint,
                gcn_hidden=4,
                gru_hidden=5,
                spatial_residual=True,
            )

    def test_spatial_residual_checkpoint_restores_architecture(self):
        model = TemporalGNN(
            2,
            3,
            np.eye(2, dtype=np.float32),
            gcn_hidden=4,
            gru_hidden=5,
            spatial_residual=True,
        )
        self.checkpoint["model_state_dict"] = model.state_dict()
        self.checkpoint["spatial_residual"] = True
        restored = load_temporal_gnn_from_checkpoint(
            self.checkpoint,
            "cpu",
            gcn_hidden=4,
            gru_hidden=5,
        )
        self.assertTrue(restored.spatial_residual)
        with torch.no_grad():
            output = restored(torch.zeros((1, 2, 2, 3)))
        self.assertEqual(tuple(output.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
