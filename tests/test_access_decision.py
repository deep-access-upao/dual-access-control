import unittest

from src.inference.decision import aggregate_scores, decide_access, is_match


class AccessDecisionTests(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        self.assertTrue(is_match(0.3128704727, 0.3128704727))
        self.assertFalse(is_match(0.3128704726, 0.3128704727))

    def test_access_requires_both_factors(self):
        expected = {
            (False, False): ("DENIED", "RFID_UNKNOWN"),
            (False, True): ("DENIED", "RFID_UNKNOWN"),
            (True, False): ("DENIED", "FACE_NO_MATCH"),
            (True, True): ("GRANTED", "RFID_AND_FACE_OK"),
        }
        for inputs, result in expected.items():
            with self.subTest(inputs=inputs):
                decision = decide_access(*inputs)
                self.assertEqual((decision.access, decision.reason), result)
                self.assertEqual(decision.granted, result[0] == "GRANTED")

    def test_aggregation_supports_max_and_mean(self):
        scores = [0.2, 0.8, 0.5]
        self.assertEqual(aggregate_scores(scores, "max"), 0.8)
        self.assertAlmostEqual(aggregate_scores(scores, "mean"), 0.5)

    def test_aggregation_rejects_empty_or_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "al menos un score"):
            aggregate_scores([], "max")
        with self.assertRaisesRegex(ValueError, "Estrategia"):
            aggregate_scores([0.5], "median")


if __name__ == "__main__":
    unittest.main()
