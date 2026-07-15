"""Interfaz principal para verificar pares y múltiples referencias faciales."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf

from src.inference.config import ModelConfig, load_model_config
from src.inference.decision import aggregate_scores, is_match
from src.inference.errors import InferenceError
from src.inference.image_preprocessing import prepare_image
from src.inference.model_loader import load_model_from_config, validate_model_signature


@dataclass(frozen=True)
class VerificationResult:
    score: float
    threshold: float
    match: bool
    decision: str
    model_name: str
    reference_image: str
    capture_image: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MultiReferenceResult:
    aggregated_score: float
    aggregation_strategy: str
    threshold: float
    match: bool
    decision: str
    model_name: str
    capture_image: str
    reference_scores: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["reference_scores"] = list(self.reference_scores)
        return payload


class FaceVerifier:
    """Verificador facial listo para CLI, backend o futura web."""

    def __init__(self, config: ModelConfig, model: Any):
        validate_model_signature(model, config.input_size)
        self.config = config
        self.model = model

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FaceVerifier":
        config = load_model_config(config_path)
        return cls(config, load_model_from_config(config))

    def _predict_scores(self, references: tf.Tensor, captures: tf.Tensor) -> list[float]:
        expected = int(references.shape[0])
        if int(captures.shape[0]) != expected:
            raise InferenceError("Los batches de referencia y captura tienen tamaños distintos.")
        try:
            raw = self.model([references, captures], training=False)
            scores = np.asarray(raw).reshape(-1)
        except Exception as error:
            raise InferenceError(f"Falló la inferencia del modelo: {error}") from error
        if len(scores) != expected:
            raise InferenceError(
                f"El modelo devolvió {len(scores)} scores para {expected} pares."
            )
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise InferenceError("El modelo devolvió scores no finitos o fuera de [0, 1].")
        return [float(score) for score in scores]

    def verify_pair(
        self, reference_image: str | Path, capture_image: str | Path
    ) -> VerificationResult:
        reference = prepare_image(reference_image, self.config.input_size)
        capture = prepare_image(capture_image, self.config.input_size)
        score = self._predict_scores(reference, capture)[0]
        match = is_match(score, self.config.threshold)
        return VerificationResult(
            score=score,
            threshold=self.config.threshold,
            match=match,
            decision=self.config.positive_label if match else self.config.negative_label,
            model_name=self.config.model_name,
            reference_image=str(reference_image),
            capture_image=str(capture_image),
        )

    def verify_against_references(
        self,
        capture_image: str | Path,
        reference_images: Iterable[str | Path],
        strategy: str = "max",
    ) -> MultiReferenceResult:
        paths = list(reference_images)
        if not paths:
            raise ValueError("Se requiere al menos una imagen de referencia.")
        if strategy not in {"max", "mean", "average"}:
            raise ValueError("Estrategia no soportada; usa 'max' o 'mean'.")

        capture = prepare_image(capture_image, self.config.input_size)
        reference_batches = [prepare_image(path, self.config.input_size) for path in paths]
        references = tf.concat(reference_batches, axis=0)
        captures = tf.repeat(capture, repeats=len(paths), axis=0)
        scores = self._predict_scores(references, captures)
        canonical_strategy = "mean" if strategy == "average" else strategy
        aggregated_score = aggregate_scores(scores, canonical_strategy)
        match = is_match(aggregated_score, self.config.threshold)
        per_reference = tuple(
            {"reference_image": str(path), "score": score}
            for path, score in zip(paths, scores)
        )
        return MultiReferenceResult(
            aggregated_score=aggregated_score,
            aggregation_strategy=canonical_strategy,
            threshold=self.config.threshold,
            match=match,
            decision=self.config.positive_label if match else self.config.negative_label,
            model_name=self.config.model_name,
            capture_image=str(capture_image),
            reference_scores=per_reference,
        )
