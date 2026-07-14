import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.callbacks import EarlyStopping
from src.training.engine import fit_with_rollout


class EarlyStoppingTest(unittest.TestCase):
    def test_strict_improvement_and_patience(self):
        callback = EarlyStopping(patience=2)
        self.assertEqual(callback.update(3.0), (True, False))
        self.assertEqual(callback.update(4.0), (False, False))
        self.assertEqual(callback.update(5.0), (False, True))
        self.assertEqual(callback.best_metric, 3.0)


class TrainingHistoryTest(unittest.TestCase):
    def test_records_learning_rate_used_before_scheduler_step(self):
        model = nn.Linear(1, 1)
        loader = DataLoader(
            TensorDataset(torch.zeros((1, 1)), torch.zeros((1, 1))),
            batch_size=1,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        class Scheduler:
            def step(self, metric):
                self.metric = metric
                optimizer.param_groups[0]["lr"] *= 0.1

        result = fit_with_rollout(
            model,
            loader,
            loader,
            nn.MSELoss(),
            optimizer,
            Scheduler(),
            "cpu",
            epochs=2,
            early_stopping=EarlyStopping(patience=3),
            rollout_metric=lambda: 1.0,
        )

        learning_rates = [record.learning_rate for record in result.records]
        self.assertAlmostEqual(learning_rates[0], 0.1)
        self.assertAlmostEqual(learning_rates[1], 0.01)
        self.assertEqual(result.best_epoch, 1)
        history = result.history_dict(epochs_requested=2)
        self.assertEqual(history["epochs_ran"], 2)
        self.assertEqual(history["stop_reason"], "max_epochs_reached")


if __name__ == "__main__":
    unittest.main()
