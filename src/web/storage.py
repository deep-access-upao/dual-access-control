"""Almacenamiento local y validado de imágenes privadas de la demo."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from src.web.schemas import ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_BYTES, ValidationError


class ImageStorage:
    def __init__(self, reference_dir: str | Path, capture_dir: str | Path):
        self.reference_dir = Path(reference_dir)
        self.capture_dir = Path(capture_dir)

    def initialize(self) -> None:
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir.mkdir(parents=True, exist_ok=True)

    async def save_reference(self, user_id: int, upload: UploadFile) -> Path:
        target_dir = self.reference_dir / str(user_id)
        return await self._save(upload, target_dir)

    async def save_capture(self, upload: UploadFile) -> Path:
        return await self._save(upload, self.capture_dir)

    async def _save(self, upload: UploadFile, directory: Path) -> Path:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
            raise ValidationError(f"Formato no permitido. Usa: {allowed}.")
        content = await upload.read(MAX_IMAGE_BYTES + 1)
        if not content:
            raise ValidationError("La imagen está vacía.")
        if len(content) > MAX_IMAGE_BYTES:
            raise ValidationError("Cada imagen debe pesar como máximo 10 MB.")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{uuid4().hex}{suffix}"
        destination.write_bytes(content)
        return destination.resolve()

    @staticmethod
    def remove(path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)
