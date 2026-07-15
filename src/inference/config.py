"""Carga y validación de la configuración versionada del modelo final."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config import INPUT_SHAPE, PROJECT_ROOT
from src.inference.errors import InvalidConfigError

REQUIRED_FIELDS = {
    "model_name",
    "model_path",
    "threshold",
    "input_size",
    "score_rule",
    "positive_label",
    "negative_label",
    "access_rule",
    "threshold_origin",
}


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    model_path: Path
    threshold: float
    input_size: tuple[int, int, int]
    score_rule: str
    positive_label: str
    negative_label: str
    access_rule: str
    threshold_origin: str
    preprocessing: Mapping[str, Any]
    source_path: Path

    @property
    def resolved_model_path(self) -> Path:
        """Resuelve la ruta versionada con relación a la raíz del proyecto."""
        return (PROJECT_ROOT / self.model_path).resolve()


def _required_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InvalidConfigError(f"El campo '{field}' debe ser texto no vacío.")
    return value.strip()


def _validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidConfigError("El campo 'threshold' debe ser numérico.")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise InvalidConfigError("El campo 'threshold' debe estar entre 0 y 1.")
    return threshold


def _validate_input_size(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise InvalidConfigError("El campo 'input_size' debe ser una lista [alto, ancho, canales].")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise InvalidConfigError("Los valores de 'input_size' deben ser enteros positivos.")
    input_size = tuple(value)
    if input_size != INPUT_SHAPE:
        raise InvalidConfigError(
            f"input_size={input_size} no coincide con el pipeline real {INPUT_SHAPE}."
        )
    return input_size


def _validate_model_path(value: str) -> Path:
    model_path = Path(value)
    if model_path.is_absolute():
        raise InvalidConfigError("'model_path' debe ser una ruta relativa al proyecto.")
    resolved = (PROJECT_ROOT / model_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise InvalidConfigError("'model_path' no puede salir de la raíz del proyecto.") from error
    if model_path.suffix.lower() != ".keras":
        raise InvalidConfigError("'model_path' debe apuntar a un artefacto .keras.")
    return model_path


def model_config_from_dict(data: Mapping[str, Any], source_path: Path) -> ModelConfig:
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise InvalidConfigError(f"Faltan campos obligatorios: {', '.join(missing)}.")

    score_rule = _required_text(data, "score_rule")
    if score_rule != "score >= threshold":
        raise InvalidConfigError("La única regla soportada es 'score >= threshold'.")

    preprocessing = data.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        raise InvalidConfigError("El campo opcional 'preprocessing' debe ser un objeto JSON.")
    expected_preprocessing = {
        "decoder": "JPEG",
        "color_space": "RGB",
        "dtype": "float32",
        "normalization": "divide_by_255",
    }
    inconsistent = {
        field: preprocessing.get(field)
        for field, expected in expected_preprocessing.items()
        if field in preprocessing and preprocessing.get(field) != expected
    }
    if inconsistent:
        raise InvalidConfigError(
            f"'preprocessing' no coincide con el pipeline real: {inconsistent}."
        )

    return ModelConfig(
        model_name=_required_text(data, "model_name"),
        model_path=_validate_model_path(_required_text(data, "model_path")),
        threshold=_validate_threshold(data.get("threshold")),
        input_size=_validate_input_size(data.get("input_size")),
        score_rule=score_rule,
        positive_label=_required_text(data, "positive_label"),
        negative_label=_required_text(data, "negative_label"),
        access_rule=_required_text(data, "access_rule"),
        threshold_origin=_required_text(data, "threshold_origin"),
        preprocessing=preprocessing,
        source_path=source_path.resolve(),
    )


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise InvalidConfigError(f"No se encontró la configuración: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidConfigError(f"No se pudo leer la configuración {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise InvalidConfigError("La raíz de la configuración debe ser un objeto JSON.")
    return model_config_from_dict(data, config_path)
