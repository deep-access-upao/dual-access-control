"""Modelos de dominio livianos para la demo web."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class User:
    id: int
    full_name: str
    rfid_uid: str
    is_active: bool
    created_at: str
    reference_count: int = 0


@dataclass(frozen=True)
class FaceReference:
    id: int
    user_id: int
    image_path: str
    created_at: str


@dataclass(frozen=True)
class AccessEvent:
    id: int
    rfid_uid: str
    user_id: Optional[int]
    user_name: Optional[str]
    score: Optional[float]
    threshold: float
    face_match: bool
    access_decision: str
    reason: str
    references_used: int
    created_at: str
