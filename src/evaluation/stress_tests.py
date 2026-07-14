"""Deterministic synthetic stress conditions for clean evaluation images."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from src.config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from src.dataset.augmentations import apply_affine_transform, as_stateless_seed

STRESS_CONDITIONS = (
    "low_light",
    "overexposure",
    "low_contrast",
    "noise",
    "blur",
    "rotation_crop",
    "partial_occlusion",
    "synthetic_glasses",
    "synthetic_beard_shadow",
)


def apply_stress_condition(
    image: tf.Tensor,
    condition: str,
    seed: int | Sequence[int] | tf.Tensor = 2026,
) -> tf.Tensor:
    """Apply one named, deterministic condition without changing source files."""
    if condition not in STRESS_CONDITIONS:
        choices = ", ".join(STRESS_CONDITIONS)
        raise ValueError(f"Unknown stress condition '{condition}'. Expected one of: {choices}")

    image = _prepare_image(image)
    seed = as_stateless_seed(seed)

    if condition == "low_light":
        stressed = tf.image.adjust_gamma(image * 0.58, gamma=1.18)
    elif condition == "overexposure":
        stressed = image * 1.30 + 0.16
    elif condition == "low_contrast":
        stressed = tf.image.adjust_contrast(image, 0.45)
    elif condition == "noise":
        noise = tf.random.stateless_normal(tf.shape(image), seed, stddev=0.055)
        stressed = image + noise
    elif condition == "blur":
        stressed = tf.nn.avg_pool2d(
            image[tf.newaxis, ...], ksize=5, strides=1, padding="SAME"
        )[0]
    elif condition == "rotation_crop":
        stressed = apply_affine_transform(
            image,
            angle_degrees=6.0,
            translate_x=3.0,
            translate_y=-2.0,
            scale=0.93,
        )
    elif condition == "partial_occlusion":
        stressed = _rectangle_overlay(
            image, x_min=0.30, x_max=0.72, y_min=0.61, y_max=0.76, alpha=0.82
        )
    elif condition == "synthetic_glasses":
        stressed = _synthetic_glasses(image)
    else:
        stressed = _synthetic_beard_shadow(image)

    stressed = tf.clip_by_value(stressed, 0.0, 1.0)
    return tf.ensure_shape(stressed, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


def build_stress_variants(
    image: tf.Tensor,
    seed: int | Sequence[int] | tf.Tensor = 2026,
) -> dict[str, tf.Tensor]:
    """Return the clean image and every stress condition as separate tensors."""
    base_seed = as_stateless_seed(seed)
    condition_seeds = tf.random.experimental.stateless_split(
        base_seed, num=len(STRESS_CONDITIONS)
    )
    variants = {"original": _prepare_image(image)}
    for index, condition in enumerate(STRESS_CONDITIONS):
        variants[condition] = apply_stress_condition(
            image, condition, seed=condition_seeds[index]
        )
    return variants


def _prepare_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return tf.ensure_shape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


def _coordinate_grid():
    y = tf.linspace(0.0, 1.0, IMAGE_HEIGHT)[:, tf.newaxis]
    x = tf.linspace(0.0, 1.0, IMAGE_WIDTH)[tf.newaxis, :]
    return x, y


def _blend_mask(image, mask, color, alpha):
    mask = tf.cast(mask, image.dtype)[..., tf.newaxis]
    opacity = mask * tf.cast(alpha, image.dtype)
    color = tf.reshape(tf.cast(color, image.dtype), [1, 1, 3])
    return image * (1.0 - opacity) + color * opacity


def _rectangle_overlay(image, x_min, x_max, y_min, y_max, alpha):
    x, y = _coordinate_grid()
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return _blend_mask(image, mask, color=(0.08, 0.08, 0.08), alpha=alpha)


def _synthetic_glasses(image):
    x, y = _coordinate_grid()
    left_lens = (x >= 0.14) & (x <= 0.43) & (y >= 0.31) & (y <= 0.48)
    right_lens = (x >= 0.57) & (x <= 0.86) & (y >= 0.31) & (y <= 0.48)
    bridge = (x >= 0.43) & (x <= 0.57) & (y >= 0.375) & (y <= 0.405)
    frame = left_lens | right_lens | bridge
    return _blend_mask(image, frame, color=(0.04, 0.04, 0.05), alpha=0.34)


def _synthetic_beard_shadow(image):
    x, y = _coordinate_grid()
    ellipse = ((x - 0.5) / 0.34) ** 2 + ((y - 0.77) / 0.24) ** 2 <= 1.0
    lower_face = ellipse & (y >= 0.59)
    return _blend_mask(image, lower_face, color=(0.10, 0.075, 0.06), alpha=0.28)
