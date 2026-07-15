"""Casos de uso de registro y decisión de acceso."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.inference.config import load_model_config
from src.inference.decision import decide_access
from src.web.database import Database
from src.web.models import User
from src.web.schemas import clean_name, normalize_rfid
from src.web.storage import ImageStorage


@dataclass(frozen=True)
class AccessOutcome:
    access_decision: str
    reason: str
    rfid_uid: str
    user: User | None
    score: float | None
    threshold: float
    face_match: bool
    references_used: int


class LazyVerifier:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._verifier: Any = None

    def __call__(self) -> Any:
        if self._verifier is None:
            from src.inference import FaceVerifier

            self._verifier = FaceVerifier.from_config(self.config_path)
        return self._verifier


class DemoService:
    def __init__(
        self,
        database: Database,
        storage: ImageStorage,
        config_path: str | Path,
        verifier_provider: Callable[[], Any] | None = None,
    ):
        self.database = database
        self.storage = storage
        self.config_path = Path(config_path)
        self.config = load_model_config(self.config_path)
        self.verifier_provider = verifier_provider or LazyVerifier(self.config_path)

    async def register_user(self, full_name: str, rfid_uid: str, uploads: list[Any]) -> User:
        name = clean_name(full_name)
        uid = normalize_rfid(rfid_uid)
        valid_uploads = [upload for upload in uploads if upload.filename]
        if not valid_uploads:
            raise ValueError("Debes agregar al menos una foto de referencia.")

        user = self.database.create_user(name, uid)
        saved: list[Path] = []
        try:
            for upload in valid_uploads:
                path = await self.storage.save_reference(user.id, upload)
                saved.append(path)
                self.database.add_reference(user.id, str(path))
        except Exception:
            self.database.delete_user(user.id)
            for path in saved:
                self.storage.remove(path)
            raise
        return self.database.get_user(user.id) or user

    def verify_access(self, rfid_uid: str, capture_path: str | Path) -> AccessOutcome:
        uid = normalize_rfid(rfid_uid)
        user = self.database.get_user_by_rfid(uid)
        threshold = self.config.threshold
        if user is None:
            return self._deny(uid, None, threshold, "RFID_UNKNOWN", 0)
        if not user.is_active:
            return self._deny(uid, user, threshold, "USER_INACTIVE", 0)

        references = self.database.list_references(user.id)
        if not references:
            return self._deny(uid, user, threshold, "NO_REFERENCES", 0)

        reference_paths = [reference.image_path for reference in references]
        try:
            result = self.verifier_provider().verify_against_references(
                capture_path, reference_paths, strategy="max"
            )
        except Exception:
            return self._deny(uid, user, threshold, "INFERENCE_ERROR", len(references))
        decision = decide_access(rfid_known=True, face_match=result.match)
        outcome = AccessOutcome(
            access_decision=decision.access,
            reason=decision.reason,
            rfid_uid=uid,
            user=user,
            score=float(result.aggregated_score),
            threshold=threshold,
            face_match=bool(result.match),
            references_used=len(references),
        )
        self._record(outcome)
        return outcome

    def _deny(
        self,
        uid: str,
        user: User | None,
        threshold: float,
        reason: str,
        references_used: int,
    ) -> AccessOutcome:
        outcome = AccessOutcome(
            access_decision="DENIED",
            reason=reason,
            rfid_uid=uid,
            user=user,
            score=None,
            threshold=threshold,
            face_match=False,
            references_used=references_used,
        )
        self._record(outcome)
        return outcome

    def _record(self, outcome: AccessOutcome) -> None:
        self.database.record_event(
            rfid_uid=outcome.rfid_uid,
            user_id=outcome.user.id if outcome.user else None,
            score=outcome.score,
            threshold=outcome.threshold,
            face_match=outcome.face_match,
            access_decision=outcome.access_decision,
            reason=outcome.reason,
            references_used=outcome.references_used,
        )
