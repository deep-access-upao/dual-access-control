import unittest

import numpy as np

from src.evaluation.evaluate import compute_metrics, select_threshold, threshold_sweep


class FormalEvaluationTests(unittest.TestCase):
    def test_threshold_is_selected_from_supplied_validation_scores(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        scores = np.array([0.05, 0.30, 0.65, 0.90], dtype=np.float32)
        search = threshold_sweep(labels, scores)
        selected = select_threshold(search)
        self.assertAlmostEqual(float(selected["f1"]), 1.0)
        self.assertGreater(float(selected["threshold"]), 0.30)
        self.assertLessEqual(float(selected["threshold"]), 0.65)

    def test_far_and_frr_follow_access_control_definitions(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        scores = np.array([0.2, 0.8, 0.4, 0.9], dtype=np.float32)
        metrics = compute_metrics(labels, scores, threshold=0.5)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(metrics["far"], 0.5)
        self.assertEqual(metrics["frr"], 0.5)


if __name__ == "__main__":
    unittest.main()
