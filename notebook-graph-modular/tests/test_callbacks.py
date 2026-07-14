import unittest

from src.training.callbacks import EarlyStopping


class EarlyStoppingTest(unittest.TestCase):
    def test_strict_improvement_and_patience(self):
        callback = EarlyStopping(patience=2)
        self.assertEqual(callback.update(3.0), (True, False))
        self.assertEqual(callback.update(4.0), (False, False))
        self.assertEqual(callback.update(5.0), (False, True))
        self.assertEqual(callback.best_metric, 3.0)


if __name__ == "__main__":
    unittest.main()

