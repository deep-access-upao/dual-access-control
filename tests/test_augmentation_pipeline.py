import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

from src.dataset import dataloader
from src.dataset.augmentations import augment_face_image
from src.evaluation.augmentation_preview import build_preview_images
from src.evaluation.stress_tests import STRESS_CONDITIONS, apply_stress_condition


class AugmentationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "face.jpg"
        self.csv_path = self.root / "pairs.csv"

        rows = tf.linspace(0.0, 1.0, 112)[:, tf.newaxis, tf.newaxis]
        columns = tf.linspace(1.0, 0.0, 112)[tf.newaxis, :, tf.newaxis]
        red = tf.broadcast_to(rows, [112, 112, 1])
        green = tf.broadcast_to(columns, [112, 112, 1])
        blue = (red + green) / 2.0
        image = tf.cast(tf.concat([red, green, blue], axis=-1) * 255.0, tf.uint8)
        tf.io.write_file(str(self.image_path), tf.io.encode_jpeg(image, quality=95))

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image_a", "image_b", "label"])
            writer.writeheader()
            writer.writerow(
                {
                    "image_a": str(self.image_path),
                    "image_b": str(self.image_path),
                    "label": 1,
                }
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_augmentation_is_reproducible_and_independent_per_branch(self):
        first = dataloader.create_pairs_dataset(
            self.csv_path,
            batch_size=1,
            shuffle=False,
            augment=True,
            seed=73,
        )
        second = dataloader.create_pairs_dataset(
            self.csv_path,
            batch_size=1,
            shuffle=False,
            augment=True,
            seed=73,
        )

        (first_a, first_b), _ = next(iter(first))
        (second_a, second_b), _ = next(iter(second))

        self.assertEqual(first_a.shape, (1, 112, 112, 3))
        self.assertEqual(first_b.shape, (1, 112, 112, 3))
        np.testing.assert_allclose(first_a.numpy(), second_a.numpy(), atol=1e-6)
        np.testing.assert_allclose(first_b.numpy(), second_b.numpy(), atol=1e-6)
        self.assertFalse(np.allclose(first_a.numpy(), first_b.numpy()))

    def test_train_wrapper_can_toggle_augmentation(self):
        with mock.patch.object(dataloader, "TRAIN_CSV", self.csv_path):
            augmented = dataloader.get_train_dataset(
                batch_size=1, augment=True, seed=31
            )
            clean = dataloader.get_train_dataset(
                batch_size=1, augment=False, seed=31
            )

        (augmented_a, _), _ = next(iter(augmented))
        (clean_a, _), _ = next(iter(clean))
        self.assertFalse(np.allclose(augmented_a.numpy(), clean_a.numpy()))

    def test_validation_and_test_wrappers_keep_images_clean(self):
        expected = dataloader.load_image(str(self.image_path)).numpy()
        with mock.patch.object(dataloader, "VAL_CSV", self.csv_path):
            validation = dataloader.get_val_dataset(batch_size=1)
        with mock.patch.object(dataloader, "TEST_CSV", self.csv_path):
            test = dataloader.get_test_dataset(batch_size=1)

        for dataset in (validation, test):
            (image_a, image_b), labels = next(iter(dataset))
            np.testing.assert_allclose(image_a[0].numpy(), expected, atol=1e-6)
            np.testing.assert_allclose(image_b[0].numpy(), expected, atol=1e-6)
            self.assertEqual(float(labels[0]), 1.0)

    def test_augmentation_preserves_model_input_shape_and_range(self):
        image = dataloader.load_image(str(self.image_path))
        augmented = augment_face_image(image, seed=[15, 90])
        self.assertEqual(augmented.shape, (112, 112, 3))
        self.assertGreaterEqual(float(tf.reduce_min(augmented)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(augmented)), 1.0)

    def test_stress_conditions_are_deterministic_and_separate(self):
        image = dataloader.load_image(str(self.image_path))
        for condition in STRESS_CONDITIONS:
            first = apply_stress_condition(image, condition, seed=[99, 7])
            second = apply_stress_condition(image, condition, seed=[99, 7])
            self.assertEqual(first.shape, (112, 112, 3))
            np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-6)
            self.assertGreaterEqual(float(tf.reduce_min(first)), 0.0)
            self.assertLessEqual(float(tf.reduce_max(first)), 1.0)

        clean_afterward = dataloader.load_image(str(self.image_path))
        np.testing.assert_allclose(image.numpy(), clean_afterward.numpy(), atol=1e-6)

    def test_preview_is_built_in_memory(self):
        image = dataloader.load_image(str(self.image_path))
        preview = build_preview_images(image, seed=2026)
        self.assertIn("original", preview)
        self.assertIn("random_train_augmentation", preview)
        self.assertIn("synthetic_glasses", preview)
        self.assertIn("synthetic_beard_shadow", preview)
        self.assertFalse((self.root / "preview.png").exists())


if __name__ == "__main__":
    unittest.main()
