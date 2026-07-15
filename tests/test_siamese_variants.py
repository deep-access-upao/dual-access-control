import unittest

import numpy as np
import tensorflow as tf

from src.models.siamese_variants import (
    build_gap_l2_cosine_model,
    build_gap_l2_embedding_network,
)


class SiameseVariantTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        self.input_shape = (32, 32, 3)

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_encoder_uses_gap_without_flatten_and_outputs_unit_embeddings(self):
        encoder = build_gap_l2_embedding_network(self.input_shape)
        layer_types = {type(layer) for layer in encoder.layers}
        self.assertIn(tf.keras.layers.GlobalAveragePooling2D, layer_types)
        self.assertNotIn(tf.keras.layers.Flatten, layer_types)
        self.assertEqual(encoder.output_shape, (None, 128))

        images = tf.random.uniform((4, *self.input_shape))
        embeddings = encoder(images, training=False).numpy()
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)

    def test_cosine_model_is_symmetric_bounded_and_serializable(self):
        model = build_gap_l2_cosine_model(self.input_shape)
        images_a = tf.random.uniform((3, *self.input_shape))
        images_b = tf.random.uniform((3, *self.input_shape))
        score_ab = model([images_a, images_b], training=False).numpy()
        score_ba = model([images_b, images_a], training=False).numpy()
        identical = model([images_a, images_a], training=False).numpy()

        np.testing.assert_allclose(score_ab, score_ba, atol=1e-6)
        self.assertTrue(np.all(score_ab >= 0.0))
        self.assertTrue(np.all(score_ab <= 1.0))
        np.testing.assert_allclose(identical, np.ones_like(identical), atol=1e-5)

        restored = tf.keras.models.model_from_json(model.to_json())
        self.assertEqual(restored.output_shape, (None, 1))


if __name__ == "__main__":
    unittest.main()
