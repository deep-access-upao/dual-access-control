"""
Valida la estructura de directorios del dataset sin leer ningún contenido multimedia.

Estructura esperada:
    data/raw/<person_id>/<view_name>/<video_file>

Códigos de salida:
    0 — los directorios raíz requeridos existen (el dataset puede estar vacío)
    1 — uno o más directorios raíz requeridos faltan
"""

import sys
from pathlib import Path

from src.config import (
    PAIRS_DIR,
    PROCESSED_DATASET_DIR,
    RAW_DATASET_DIR,
    SUPPORT_SET_DIR,
    SUPPORTED_FACE_VIEWS,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

ROOT_DIRS = {
    "data/raw": RAW_DATASET_DIR,
    "data/processed": PROCESSED_DATASET_DIR,
    "data/pairs": PAIRS_DIR,
    "data/support_set": SUPPORT_SET_DIR,
}


def validate_root_directories() -> bool:
    """Verifica que todos los directorios raíz requeridos existan. Retorna True si todos están presentes."""
    print("=== Root Directory Check ===")
    all_present = True
    for label, path in ROOT_DIRS.items():
        status = "OK" if path.is_dir() else "MISSING"
        print(f"  [{status}] {label}")
        if status == "MISSING":
            all_present = False
    print()
    return all_present


def get_person_directories() -> list[Path]:
    """Retorna la lista ordenada de subdirectorios de personas dentro de data/raw."""
    return sorted(
        entry for entry in RAW_DATASET_DIR.iterdir() if entry.is_dir()
    )


def inspect_person_views(person_dir: Path) -> dict[str, int]:
    """
    Retorna un mapeo de view_name -> cantidad de videos para un directorio de persona.
    Solo cuenta archivos con extensiones de video soportadas.
    """
    view_counts: dict[str, int] = {}
    for view in SUPPORTED_FACE_VIEWS:
        view_path = person_dir / view
        if view_path.is_dir():
            count = sum(
                1
                for f in view_path.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            view_counts[view] = count
    return view_counts


def print_report(person_dirs: list[Path]) -> None:
    """Imprime un reporte por persona con las vistas presentes y la cantidad de videos."""
    print("=== Dataset Report ===")
    for person_dir in person_dirs:
        print(f"\n  Person: {person_dir.name}")
        view_counts = inspect_person_views(person_dir)

        present = [v for v in SUPPORTED_FACE_VIEWS if v in view_counts]
        missing = [v for v in SUPPORTED_FACE_VIEWS if v not in view_counts]

        if present:
            print("    Present views:")
            for view in present:
                print(f"      {view}: {view_counts[view]} video(s)")
        else:
            print("    Present views: none")

        if missing:
            print(f"    Missing views: {', '.join(missing)}")
        else:
            print("    Missing views: none")
    print()


def main() -> None:
    all_roots_present = validate_root_directories()

    if not all_roots_present:
        print("ERROR: Required root directories are missing. Create them before populating the dataset.")
        sys.exit(1)

    person_dirs = get_person_directories()

    if not person_dirs:
        print("Dataset is empty: data/raw exists but contains no person folders.")
        sys.exit(0)

    print_report(person_dirs)
    print(f"Validation complete. {len(person_dirs)} person(s) found.")


if __name__ == "__main__":
    main()
