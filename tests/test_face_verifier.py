import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tensorflow as tf

from src.inference.config import load_model_config
from src.inference.errors import ImageNotFoundError, ModelNotFoundError
from src.inference.face_verifier import FaceVerifier
from src.inference.image_preprocessing import prepare_image
from src.inference.model_loader import load_model_from_config


class FakeSiameseModel:
    inputs = [
        SimpleNamespace(shape=(None, 112, 112, 3)),
        SimpleNamespace(shape=(None, 112, 112, 3)),
    ]
    output_shape = (None, 1)

    def __call__(self, inputs, training=False):
        del training
        count = int(inputs[0].shape[0])
        return np.asarray([[0.2], [0.7], [0.4]], dtype=np.float32)[:count]


class FaceVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_model_config("config/model_config.json")

    def test_multiple_references_return_individual_and_max_scores(self):
        verifier = FaceVerifier(self.config, FakeSiameseModel())
        prepared = tf.zeros((1, 112, 112, 3), dtype=tf.float32)
        with mock.patch("src.inference.face_verifier.prepare_image", return_value=prepared):
            result = verifier.verify_against_references(
                "capture.jpg", ["front.jpg", "left.jpg", "right.jpg"]
            )
        self.assertAlmostEqual(result.aggregated_score, 0.7, places=6)
        self.assertEqual(result.decision, "MATCH")
        self.assertEqual(len(result.reference_scores), 3)

    def test_pair_result_uses_frozen_threshold(self):
        verifier = FaceVerifier(self.config, FakeSiameseModel())
        prepared = tf.zeros((1, 112, 112, 3), dtype=tf.float32)
        with mock.patch("src.inference.face_verifier.prepare_image", return_value=prepared):
            result = verifier.verify_pair("reference.jpg", "capture.jpg")
        self.assertEqual(result.score, np.float32(0.2))
        self.assertFalse(result.match)
        self.assertEqual(result.decision, "NO_MATCH")
        self.assertEqual(result.threshold, 0.3128704727)

    def test_missing_image_has_clear_error(self):
        with self.assertRaisesRegex(ImageNotFoundError, "No se encontró la imagen"):
            prepare_image("missing-image.jpg")

    def test_missing_model_has_clear_error_without_loading_tensorflow_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            payload = json.loads(Path("config/model_config.json").read_text(encoding="utf-8"))
            payload["model_path"] = "models/saved_model/missing-for-test.keras"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_model_config(config_path)
            with self.assertRaisesRegex(ModelNotFoundError, "No se encontró el modelo final"):
                load_model_from_config(config)


if __name__ == "__main__":
    unittest.main()
