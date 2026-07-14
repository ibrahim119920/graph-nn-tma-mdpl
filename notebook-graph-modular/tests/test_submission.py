import unittest

import numpy as np
import pandas as pd

from src.inference.submission import (
    build_submission,
    predictions_to_frame,
    summarize_submission,
)


class SubmissionTest(unittest.TestCase):
    def test_submission_preserves_sample_id_order(self):
        first = pd.Timestamp("2025-09-19 06:00:00")
        second = pd.Timestamp("2025-09-19 12:00:00")
        sample = pd.DataFrame(
            {
                "id": [
                    f"{second:%Y-%m-%d %H:%M:%S} - B",
                    f"{first:%Y-%m-%d %H:%M:%S} - A",
                    f"{second:%Y-%m-%d %H:%M:%S} - A",
                    f"{first:%Y-%m-%d %H:%M:%S} - B",
                ],
                "datetime": [second, first, second, first],
                "nama_pos": ["B", "A", "A", "B"],
            }
        )
        frame = predictions_to_frame(
            {
                first: np.array([1.0, 2.0]),
                second: np.array([3.0, 4.0]),
            },
            ["A", "B"],
        )
        submission, missing = build_submission(sample, frame)
        self.assertEqual(missing, 0)
        self.assertEqual(submission["id"].tolist(), sample["id"].tolist())
        self.assertEqual(submission["tma_mdpl"].tolist(), [4.0, 1.0, 3.0, 2.0])

    def test_prediction_width_must_match_node_order(self):
        with self.assertRaisesRegex(ValueError, "node_order"):
            predictions_to_frame(
                {pd.Timestamp("2025-01-01"): np.array([1.0])},
                ["A", "B"],
            )

    def test_summary_reports_submission_validation_and_distribution(self):
        submission = pd.DataFrame(
            {
                "id": ["one", "two", "three"],
                "tma_mdpl": [1.0, 2.0, 4.0],
            }
        )
        summary = summarize_submission(submission)
        self.assertEqual(summary["columns"], ["id", "tma_mdpl"])
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["id_unique"], 3)
        self.assertEqual(summary["prediction_unique_values"], 3)
        self.assertEqual(summary["prediction_min"], 1.0)
        self.assertEqual(summary["prediction_max"], 4.0)


if __name__ == "__main__":
    unittest.main()
