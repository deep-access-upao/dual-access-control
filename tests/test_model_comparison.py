import unittest

from src.evaluation.compare_model_variants import (
    BASELINE_NAME,
    PHOTOMETRIC_CONDITIONS,
    VARIANT_NAME,
    recommend_model,
)


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "validation_far": 0.02,
            "validation_f1": 0.965,
            "validation_accuracy": 0.966,
        }
        self.variant = {
            "validation_far": 0.02,
            "validation_f1": 0.960,
            "validation_accuracy": 0.960,
        }
        self.baseline_stress = {
            condition: {"frr": 0.30, "far": 0.02}
            for condition in PHOTOMETRIC_CONDITIONS
        }
        self.variant_stress = {
            condition: {"frr": 0.24, "far": 0.03}
            for condition in PHOTOMETRIC_CONDITIONS
        }

    def test_variant_replaces_baseline_only_when_all_controls_pass(self):
        decision = recommend_model(
            self.baseline,
            self.variant,
            self.baseline_stress,
            self.variant_stress,
            validation_stress_available=True,
        )
        self.assertTrue(decision["replace_baseline"])
        self.assertEqual(decision["selected_model"], VARIANT_NAME)

    def test_clean_far_regression_keeps_baseline(self):
        self.variant["validation_far"] = 0.024
        decision = recommend_model(
            self.baseline,
            self.variant,
            self.baseline_stress,
            self.variant_stress,
            validation_stress_available=True,
        )
        self.assertFalse(decision["replace_baseline"])
        self.assertEqual(decision["selected_model"], BASELINE_NAME)

    def test_small_photometric_gain_keeps_baseline(self):
        for condition in PHOTOMETRIC_CONDITIONS:
            self.variant_stress[condition]["frr"] = 0.27
        decision = recommend_model(
            self.baseline,
            self.variant,
            self.baseline_stress,
            self.variant_stress,
            validation_stress_available=True,
        )
        self.assertFalse(decision["replace_baseline"])
        self.assertEqual(
            decision["secondary_test_stress"][
                "photometric_frr_improvement_at_least_five_points"
            ],
            [],
        )

    def test_missing_validation_stress_keeps_baseline(self):
        decision = recommend_model(
            self.baseline,
            self.variant,
            self.baseline_stress,
            self.variant_stress,
            validation_stress_available=False,
        )
        self.assertFalse(decision["replace_baseline"])
        self.assertFalse(decision["criteria"]["target_improvement_validated"])


if __name__ == "__main__":
    unittest.main()
