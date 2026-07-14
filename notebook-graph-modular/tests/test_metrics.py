import unittest

import numpy as np

from src.evaluation.metrics import mae, normalized_rmse, per_node_rmse, rmse


class MetricsTest(unittest.TestCase):
    def test_metrics_match_direct_formulas(self):
        prediction = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        target = np.array([[0.0, 2.0], [2.0, 5.0]], dtype=np.float32)
        scale = np.array([2.0, 4.0], dtype=np.float32)
        self.assertEqual(
            rmse(prediction, target),
            float(np.sqrt(np.mean((prediction - target) ** 2))),
        )
        self.assertEqual(
            mae(prediction, target),
            float(np.mean(np.abs(prediction - target))),
        )
        np.testing.assert_array_equal(
            per_node_rmse(prediction, target),
            np.sqrt(np.mean((prediction - target) ** 2, axis=0)),
        )
        self.assertEqual(
            normalized_rmse(prediction, target, scale),
            float(
                np.sqrt(
                    np.mean(((prediction - target) / scale) ** 2)
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()

