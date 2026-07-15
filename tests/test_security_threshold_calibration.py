import unittest

import numpy as np

from src.evaluation.calibrate_security_thresholds import (
    build_security_search,
    select_security_candidate,
)


class SecurityThresholdCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        self.scores = np.array(
            [0.05, 0.20, 0.40, 0.80, 0.10, 0.55, 0.75, 0.95],
            dtype=np.float32,
        )
        self.search = build_security_search(self.labels, self.scores)

    def test_balanced_far_frr_minimizes_the_gap(self):
        selected, _ = select_security_candidate(self.search, "balanced_far_frr")
        self.assertIsNotNone(selected)
        self.assertEqual(
            float(selected["far_frr_gap"]), float(self.search["far_frr_gap"].min())
        )

    def test_far_constraint_is_applied_to_validation_search(self):
        selected, _ = select_security_candidate(self.search, "far_lte_1")
        self.assertIsNotNone(selected)
        self.assertLessEqual(float(selected["far"]), 0.01)

    def test_security_first_respects_far_and_frr_operating_limits(self):
        labels = np.array([0] * 100 + [1] * 100, dtype=np.int32)
        scores = np.concatenate(
            (
                np.linspace(0.0, 0.19, 100),
                np.linspace(0.18, 1.0, 100),
            )
        )
        search = build_security_search(labels, scores)
        selected, comment = select_security_candidate(search, "security_first")
        self.assertIsNotNone(selected)
        self.assertLessEqual(float(selected["far"]), 0.02)
        self.assertLessEqual(float(selected["frr"]), 0.05)
        self.assertIn("FAR <= 2%", comment)

    def test_unknown_criterion_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Criterio no soportado"):
            select_security_candidate(self.search, "test_driven")


if __name__ == "__main__":
    unittest.main()
