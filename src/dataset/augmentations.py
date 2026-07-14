"""Realistic, moderate and reproducible augmentations for face images.

The functions in this module operate on normalized RGB tensors and do not write
images to disk. Randomness is stateless so callers can reproduce a sample by
reusing the same two-integer seed.
"""

from __future__ import annotations

import math
from typing import Sequence

import tensorflow as tf

from src.config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH

DEFAULT_AUGMENTATION_SEED = 42


def as_stateless_seed(seed: int | Sequence[int] | tf.Tensor) -> tf.Tensor:
    """Return a seed with the ``[2]`` shape expected by stateless TF ops."""
    seed_tensor = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed_tensor.shape.rank == 0:
        seed_tensor = tf.stack([seed_tensor, seed_tensor + 1])
    return tf.ensure_shape(seed_tensor, [2])


def apply_affine_transform(
    image: tf.Tensor,
    *,
    angle_degrees: tf.Tensor | float = 0.0,
    translate_x: tf.Tensor | float = 0.0,
    translate_y: tf.Tensor | float = 0.0,
    scale: tf.Tensor | float = 1.0,
) -> tf.Tensor:
    """Apply a centered affine transform while preserving the image shape."""
    image = _prepare_image(image)
    dtype = image.dtype
    angle = tf.cast(angle_degrees, dtype) * tf.cast(math.pi / 180.0, dtype)
    scale = tf.cast(scale, dtype)
    translate_x = tf.cast(translate_x, dtype)
    translate_y = tf.cast(translate_y, dtype)

    cosine = tf.math.cos(angle) * scale
    sine = tf.math.sin(angle) * scale
    center_x = tf.cast(IMAGE_WIDTH - 1, dtype) / 2.0
    center_y = tf.cast(IMAGE_HEIGHT - 1, dtype) / 2.0

    offset_x = (1.0 - cosine) * center_x + sine * center_y - translate_x
    offset_y = -sine * center_x + (1.0 - cosine) * center_y - translate_y
    transform = tf.stack(
        [cosine, -sine, offset_x, sine, cosine, offset_y, 0.0, 0.0]
    )

    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transform[tf.newaxis, ...],
        output_shape=tf.constant([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return tf.ensure_shape(transformed, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


def augment_face_image(
    image: tf.Tensor,
    seed: int | Sequence[int] | tf.Tensor = DEFAULT_AUGMENTATION_SEED,
) -> tf.Tensor:
    """Apply moderate identity-preserving augmentation to one face image.

    The caller should supply different seeds for the two Siamese branches.
    Geometry and illumination are always varied slightly. Noise, blur, mild
    resolution degradation and cutout are sampled independently.
    """
    image = _prepare_image(image)
    seeds = tf.random.experimental.stateless_split(as_stateless_seed(seed), num=16)

    angle = tf.random.stateless_uniform([], seeds[0], minval=-7.0, maxval=7.0)
    translate_x = tf.random.stateless_uniform(
        [], seeds[1], minval=-0.04 * IMAGE_WIDTH, maxval=0.04 * IMAGE_WIDTH
    )
    translate_y = tf.random.stateless_uniform(
        [], seeds[2], minval=-0.04 * IMAGE_HEIGHT, maxval=0.04 * IMAGE_HEIGHT
    )
    scale = tf.random.stateless_uniform([], seeds[3], minval=0.94, maxval=1.06)
    image = apply_affine_transform(
        image,
        angle_degrees=angle,
        translate_x=translate_x,
        translate_y=translate_y,
        scale=scale,
    )

    brightness_delta = tf.random.stateless_uniform(
        [], seeds[4], minval=-0.12, maxval=0.12
    )
    contrast_factor = tf.random.stateless_uniform(
        [], seeds[5], minval=0.85, maxval=1.15
    )
    gamma = tf.random.stateless_uniform([], seeds[6], minval=0.90, maxval=1.10)
    image = tf.image.adjust_brightness(image, brightness_delta)
    image = tf.image.adjust_contrast(image, contrast_factor)
    image = tf.image.adjust_gamma(tf.clip_by_value(image, 0.0, 1.0), gamma=gamma)

    noise_sigma = tf.random.stateless_uniform(
        [], seeds[8], minval=0.005, maxval=0.025
    )
    image = _apply_with_probability(
        image,
        seeds[7],
        probability=0.35,
        transform=lambda value: value
        + tf.random.stateless_normal(tf.shape(value), seeds[9], stddev=noise_sigma),
    )
    image = _apply_with_probability(
        image, seeds[10], probability=0.20, transform=lambda value: _box_blur(value, 3)
    )
    image = _apply_with_probability(
        image,
        seeds[11],
        probability=0.15,
        transform=lambda value: _mild_resolution_degradation(value, seeds[12]),
    )
    image = _apply_with_probability(
        image,
        seeds[13],
        probability=0.25,
        transform=lambda value: _random_cutout(value, seeds[14], seeds[15]),
    )

    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.ensure_shape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


def _prepare_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return tf.ensure_shape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


def _apply_with_probability(image, seed, probability, transform):
    should_apply = tf.random.stateless_uniform([], seed) < probability
    return tf.cond(should_apply, lambda: transform(image), lambda: image)


def _box_blur(image: tf.Tensor, kernel_size: int) -> tf.Tensor:
    blurred = tf.nn.avg_pool2d(
        image[tf.newaxis, ...],
        ksize=kernel_size,
        strides=1,
        padding="SAME",
    )[0]
    return tf.ensure_shape(blurred, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


def _mild_resolution_degradation(image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    factor = tf.random.stateless_uniform([], seed, minval=0.76, maxval=0.94)
    small_height = tf.maximum(64, tf.cast(IMAGE_HEIGHT * factor, tf.int32))
    small_width = tf.maximum(64, tf.cast(IMAGE_WIDTH * factor, tf.int32))
    small = tf.image.resize(image, [small_height, small_width], method="area")
    return tf.image.resize(small, [IMAGE_HEIGHT, IMAGE_WIDTH], method="bilinear")


def _random_cutout(
    image: tf.Tensor, size_seed: tf.Tensor, position_seed: tf.Tensor
) -> tf.Tensor:
    size_seeds = tf.random.experimental.stateless_split(size_seed, num=2)
    position_seeds = tf.random.experimental.stateless_split(position_seed, num=2)
    cutout_height = tf.cast(
        tf.random.stateless_uniform([], size_seeds[0], minval=0.08, maxval=0.18)
        * IMAGE_HEIGHT,
        tf.int32,
    )
    cutout_width = tf.cast(
        tf.random.stateless_uniform([], size_seeds[1], minval=0.08, maxval=0.18)
        * IMAGE_WIDTH,
        tf.int32,
    )
    top = tf.random.stateless_uniform(
        [],
        position_seeds[0],
        minval=0,
        maxval=IMAGE_HEIGHT - cutout_height + 1,
        dtype=tf.int32,
    )
    left = tf.random.stateless_uniform(
        [],
        position_seeds[1],
        minval=0,
        maxval=IMAGE_WIDTH - cutout_width + 1,
        dtype=tf.int32,
    )
    rows = tf.range(IMAGE_HEIGHT)[:, tf.newaxis]
    columns = tf.range(IMAGE_WIDTH)[tf.newaxis, :]
    inside = (
        (rows >= top)
        & (rows < top + cutout_height)
        & (columns >= left)
        & (columns < left + cutout_width)
    )
    mask = tf.cast(~inside, image.dtype)[..., tf.newaxis]
    fill = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    return image * mask + fill * (1.0 - mask)
