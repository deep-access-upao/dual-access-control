"""Preprocesamiento de inferencia idéntico al usado por los datasets del modelo."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.config import INPUT_SHAPE
from src.dataset.dataloader import load_image
from src.inference.errors import ImageNotFoundError, InvalidImageError


def prepare_image(
    image_path: str | Path,
    expected_input_size: tuple[int, int, int] = INPUT_SHAPE,
) -> tf.Tensor:
    """Lee un JPEG como RGB, redimensiona, normaliza y agrega dimensión batch."""
    path = Path(image_path)
    if not path.is_file():
        raise ImageNotFoundError(f"No se encontró la imagen: {path}")
    if tuple(expected_input_size) != INPUT_SHAPE:
        raise InvalidImageError(
            f"El preprocesamiento disponible produce {INPUT_SHAPE}, no {expected_input_size}."
        )
    try:
        image = load_image(str(path))
    except (tf.errors.OpError, ValueError) as error:
        raise InvalidImageError(
            f"No se pudo decodificar '{path}' como JPEG RGB válido: {error}"
        ) from error
    if tuple(image.shape) != INPUT_SHAPE:
        raise InvalidImageError(
            f"Shape incompatible para '{path}': se obtuvo {tuple(image.shape)}; "
            f"se esperaba {INPUT_SHAPE}."
        )
    if image.dtype != tf.float32:
        raise InvalidImageError(
            f"Dtype incompatible para '{path}': {image.dtype}; se esperaba float32."
        )
    return tf.expand_dims(image, axis=0)
