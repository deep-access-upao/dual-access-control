"""Validaciones compartidas por los formularios web."""

from __future__ import annotations

import re


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
RFID_PATTERN = re.compile(r"^[A-Z0-9:_-]{2,64}$")


class ValidationError(ValueError):
    """Error de entrada que puede mostrarse al usuario."""


def clean_name(value: str) -> str:
    name = " ".join(value.split())
    if not name:
        raise ValidationError("El nombre es obligatorio.")
    if len(name) > 120:
        raise ValidationError("El nombre no puede superar 120 caracteres.")
    return name


def normalize_rfid(value: str) -> str:
    uid = value.strip().upper().replace(" ", "")
    if not uid:
        raise ValidationError("El UID RFID es obligatorio.")
    if not RFID_PATTERN.fullmatch(uid):
        raise ValidationError(
            "El UID RFID debe tener entre 2 y 64 caracteres alfanuméricos; "
            "también admite ':', '-' y '_'."
        )
    return uid
