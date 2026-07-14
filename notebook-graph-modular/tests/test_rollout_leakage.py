import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.rollout import RolloutValidationContext, evaluate_rollout
from src.inference.rollout import AutoregressiveFeatureSpec


class PersistenceModel(nn.Module):
    def forward(self, features):
        return features[:, -1, :, 0]


class RolloutLeakageTest(unittest.TestCase):
    def test_gap_is_predicted_without_resetting_actual_validation(self):
        datetimes = pd.date_range("2025-01-01", periods=6, freq="6h")
        actual = np.array([0.0, 1.0, 2.0, 30.0, 40.0, 50.0])
        panel = actual.reshape(-1, 1, 1).astype(np.float32)
        context = RolloutValidationContext(
            panel=panel,
            observation_datetimes=datetimes,
            training_datetimes=pd.DatetimeIndex(
                [datetimes[2], datetimes[4]]
            ),
            validation_start_index=0,
            time_window=2,
            feature_spec=AutoregressiveFeatureSpec(
                water_level_position=0,
                lag_positions=[],
                rolling_positions={},
            ),
            station_center=np.array([0.0], dtype=np.float32),
            station_scale=np.array([1.0], dtype=np.float32),
            station_low=np.array([-100.0], dtype=np.float32),
            station_high=np.array([100.0], dtype=np.float32),
            normalize_features=lambda values: values,
        )
        metric = evaluate_rollout(PersistenceModel(), context, "cpu")

        # Model harus meneruskan prediksi 2.0 melewati gap posisi 3/4.
        expected = float(np.sqrt(((2.0 - 30.0) ** 2 + (2.0 - 50.0) ** 2) / 2))
        self.assertAlmostEqual(metric, expected, places=5)


if __name__ == "__main__":
    unittest.main()
