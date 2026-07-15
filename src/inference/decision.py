"""Reglas deterministas de verificación y control de acceso dual."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


def _valid_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Score inválido: {value}; se esperaba un valor entre 0 y 1.")
    return score


def is_match(score: float, threshold: float) -> bool:
    checked_score = _valid_score(score)
    checked_threshold = _valid_score(threshold)
    return checked_score >= checked_threshold


def aggregate_scores(scores: Iterable[float], strategy: str = "max") -> float:
    values = [_valid_score(score) for score in scores]
    if not values:
        raise ValueError("Se requiere al menos un score de referencia.")
    if strategy == "max":
        return max(values)
    if strategy in {"mean", "average"}:
        return sum(values) / len(values)
    raise ValueError("Estrategia no soportada; usa 'max' o 'mean'.")


@dataclass(frozen=True)
class AccessDecision:
    access: str
    granted: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def decide_access(rfid_known: bool, face_match: bool) -> AccessDecision:
    if not rfid_known:
        return AccessDecision(access="DENIED", granted=False, reason="RFID_UNKNOWN")
    if not face_match:
        return AccessDecision(access="DENIED", granted=False, reason="FACE_NO_MATCH")
    return AccessDecision(access="GRANTED", granted=True, reason="RFID_AND_FACE_OK")
