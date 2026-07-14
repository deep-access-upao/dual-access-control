"""Valida referencias del support set y evita reutilizar imágenes de train."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

from src.config import DATASET_MANIFEST_PATH, SUPPORT_REFERENCE_VIEWS, SUPPORT_SET_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REFERENCE_PATTERN = re.compile(
    rf"^({'|'.join(SUPPORT_REFERENCE_VIEWS)})(?:_\d+)?$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida el support set y su separación de train.")
    parser.add_argument("--strict", action="store_true", help="Falla si el support set está vacío.")
    parser.add_argument("--manifest", type=Path, default=DATASET_MANIFEST_PATH)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_hashes(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        return {
            row["sha256"] for row in csv.DictReader(file)
            if row.get("status") == "usable" and row.get("split") == "train"
        }


def validate_support_set(strict: bool, manifest_path: Path = DATASET_MANIFEST_PATH) -> int:
    print("=== Validación del support set ===")
    if not SUPPORT_SET_DIR.is_dir():
        print(f"ERROR: no existe {SUPPORT_SET_DIR}")
        return 1

    user_dirs = sorted(path for path in SUPPORT_SET_DIR.iterdir() if path.is_dir())
    unexpected_root = [path for path in SUPPORT_SET_DIR.iterdir() if path.is_file() and path.name != ".gitkeep"]
    if not user_dirs:
        print("Support set vacío: preparado para una o más referencias por usuario.")
        return 1 if strict or unexpected_root else 0

    train_hashes = training_hashes(manifest_path)
    errors: list[str] = []
    total_references = 0
    for user_dir in user_dirs:
        references: list[Path] = []
        for path in sorted(user_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not REFERENCE_PATTERN.match(path.stem):
                errors.append(f"{user_dir.name}/{path.name}: nombre o extensión no soportada")
                continue
            references.append(path)
            if sha256(path) in train_hashes:
                errors.append(f"{user_dir.name}/{path.name}: reutiliza una imagen de entrenamiento")
        if not references:
            errors.append(f"{user_dir.name}: no contiene referencias válidas")
        total_references += len(references)
        print(f"{user_dir.name}: {len(references)} referencia(s)")

    if unexpected_root:
        errors.append("hay archivos inesperados en la raíz de data/support_set")
    print(f"Usuarios: {len(user_dirs)}; referencias: {total_references}")
    if errors:
        print("VALIDACIÓN FALLIDA")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("VALIDACIÓN APROBADA: ninguna referencia coincide con train.")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(validate_support_set(args.strict, args.manifest))


if __name__ == "__main__":
    main()
