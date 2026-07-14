"""Small visualization helpers for augmentation and synthetic stress checks."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import tensorflow as tf

from src.dataset.augmentations import augment_face_image
from src.dataset.dataloader import load_image
from src.evaluation.stress_tests import build_stress_variants


def build_preview_images(image: tf.Tensor, seed: int = 2026) -> dict[str, tf.Tensor]:
    """Build an in-memory preview; no files are created or modified."""
    variants = build_stress_variants(image, seed=[seed, 0])
    preview = {
        "original": variants["original"],
        "random_train_augmentation": augment_face_image(image, seed=[seed, 1]),
    }
    preview.update(variants)
    return preview


def preview_from_path(image_path, seed: int = 2026) -> dict[str, tf.Tensor]:
    """Load one image using the project loader and return preview tensors."""
    return build_preview_images(load_image(str(image_path)), seed=seed)


def plot_preview(
    images: dict[str, tf.Tensor], columns: int = 4, figsize=(13, 9)
):
    """Plot preview tensors and return the Matplotlib figure."""
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for axis, (name, image) in zip(axes, images.items()):
        axis.imshow(tf.clip_by_value(image, 0.0, 1.0).numpy())
        axis.set_title(name.replace("_", " "))
        axis.axis("off")
    for axis in axes[len(images) :]:
        axis.axis("off")
    figure.tight_layout()
    return figure
