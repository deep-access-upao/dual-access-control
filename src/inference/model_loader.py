"""Carga y comprobación ligera del artefacto Keras final."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from src.inference.config import ModelConfig
from src.inference.errors import ModelLoadError, ModelNotFoundError
from src.models.siamese_network import l1_distance


def _shape_tuple(value: Any) -> tuple[Any, ...]:
    shape = getattr(value, "shape", value)
    if hasattr(shape, "as_list"):
        shape = shape.as_list()
    return tuple(shape)


def model_input_shapes(model: Any) -> list[tuple[Any, ...]]:
    inputs = getattr(model, "inputs", None)
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        raise ModelLoadError("El modelo debe tener exactamente dos entradas de imagen.")
    return [_shape_tuple(item) for item in inputs]


def validate_model_signature(model: Any, expected_input_size: tuple[int, int, int]) -> None:
    shapes = model_input_shapes(model)
    incompatible = [shape for shape in shapes if len(shape) != 4 or tuple(shape[1:]) != expected_input_size]
    if incompatible:
        raise ModelLoadError(
            f"Shape de entrada incompatible: {shapes}; se esperaba "
            f"[(None, {expected_input_size[0]}, {expected_input_size[1]}, {expected_input_size[2]})] x2."
        )
    output_shape = _shape_tuple(getattr(model, "output_shape", (None, 1)))
    if len(output_shape) != 2 or output_shape[-1] != 1:
        raise ModelLoadError(
            f"Shape de salida incompatible: {output_shape}; se esperaba (None, 1)."
        )


def load_keras_model(model_path: str | Path) -> tf.keras.Model:
    path = Path(model_path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"No se encontró el modelo final: {path}. "
            "Coloca el artefacto .keras en la ruta configurada; no se descarga automáticamente."
        )
    try:
        return tf.keras.models.load_model(
            str(path), custom_objects={"l1_distance": l1_distance}, compile=False
        )
    except Exception as error:
        raise ModelLoadError(f"Keras no pudo cargar el modelo '{path}': {error}") from error


def load_model_from_config(config: ModelConfig) -> tf.keras.Model:
    model = load_keras_model(config.resolved_model_path)
    validate_model_signature(model, config.input_size)
    return model


def check_model_artifact(config: ModelConfig) -> dict[str, Any]:
    model = load_model_from_config(config)
    return {
        "status": "OK",
        "model_name": config.model_name,
        "model_path": config.model_path.as_posix(),
        "threshold": config.threshold,
        "input_size": list(config.input_size),
        "model_input_shapes": [list(shape) for shape in model_input_shapes(model)],
        "model_output_shape": list(_shape_tuple(model.output_shape)),
    }
