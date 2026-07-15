import json
import tempfile
import unittest
from pathlib import Path

from src.inference.config import load_model_config
from src.inference.errors import InvalidConfigError


class InferenceConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "model_config.json"
        self.payload = {
            "model_name": "baseline_formal/baseline_con_aumento",
            "model_path": "models/saved_model/baseline_formal/baseline_con_aumento.keras",
            "threshold": 0.3128704727,
            "input_size": [112, 112, 3],
            "score_rule": "score >= threshold",
            "positive_label": "MATCH",
            "negative_label": "NO_MATCH",
            "access_rule": "RFID conocido AND rostro verificado => GRANTED",
            "threshold_origin": "Validation, sesión 4C, security_first.",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_loads_versioned_final_configuration(self):
        self.write()
        config = load_model_config(self.path)
        self.assertEqual(config.input_size, (112, 112, 3))
        self.assertEqual(config.threshold, 0.3128704727)
        self.assertFalse(config.model_path.is_absolute())

    def test_rejects_threshold_outside_probability_range(self):
        for invalid in (-0.01, 1.01, float("inf"), True):
            with self.subTest(invalid=invalid):
                self.payload["threshold"] = invalid
                self.write()
                with self.assertRaisesRegex(InvalidConfigError, "threshold"):
                    load_model_config(self.path)

    def test_rejects_input_shape_inconsistent_with_training_pipeline(self):
        self.payload["input_size"] = [224, 224, 3]
        self.write()
        with self.assertRaisesRegex(InvalidConfigError, "pipeline real"):
            load_model_config(self.path)

    def test_rejects_absolute_model_path(self):
        self.payload["model_path"] = str((Path(self.temp_dir.name) / "model.keras").resolve())
        self.write()
        with self.assertRaisesRegex(InvalidConfigError, "ruta relativa"):
            load_model_config(self.path)

    def test_rejects_declared_bgr_pipeline(self):
        self.payload["preprocessing"] = {"color_space": "BGR"}
        self.write()
        with self.assertRaisesRegex(InvalidConfigError, "pipeline real"):
            load_model_config(self.path)


if __name__ == "__main__":
    unittest.main()
