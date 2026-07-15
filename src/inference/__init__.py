"""API reutilizable de inferencia para el modelo facial final."""

from src.inference.config import ModelConfig, load_model_config
from src.inference.decision import AccessDecision, decide_access
from src.inference.face_verifier import (
    FaceVerifier,
    MultiReferenceResult,
    VerificationResult,
)

__all__ = [
    "AccessDecision",
    "FaceVerifier",
    "ModelConfig",
    "MultiReferenceResult",
    "VerificationResult",
    "decide_access",
    "load_model_config",
]
