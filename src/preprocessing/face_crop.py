"""
Detección y recorte de caras desde fotogramas extraídos.

Input:  data/processed/<person_id>/<view_name>/frames/<frame_file>.jpg
Output: data/processed/<person_id>/<view_name>/faces/<face_file>.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    PROCESSED_DATASET_DIR,
    SUPPORTED_FACE_VIEWS,
    IMAGE_SIZE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MARGIN = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detectar y recortar rostros de los fotogramas extraídos."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobreescribir archivos de caras existentes.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Margen alrededor del recorte facial como fracción (default: {DEFAULT_MARGIN})",
    )
    return parser.parse_args()


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar el clasificador Haar: {cascade_path}")
    return detector


def find_frame_files(processed_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Busca fotogramas con estructura <person_id>/<view_name>/frames/<file>.
    Retorna lista de (person_id, view_name, frame_path).
    """
    entries: list[tuple[str, str, Path]] = []

    for person_dir in sorted(processed_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for view_dir in sorted(person_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            if view_dir.name not in SUPPORTED_FACE_VIEWS:
                print(f"  [skip] Vista no reconocida: {view_dir.relative_to(processed_dir)}")
                continue
            frames_dir = view_dir / "frames"
            if not frames_dir.is_dir():
                continue
            for frame_file in sorted(frames_dir.iterdir()):
                if frame_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    entries.append((person_dir.name, view_dir.name, frame_file))

    return entries


def crop_with_margin(
    image: np.ndarray, x: int, y: int, w: int, h: int, margin: float
) -> np.ndarray:
    """
    Recorta la cara ampliando el bounding box por la fracción indicada.
    Clampea coordenadas para no salir del límite de la imagen.
    """
    img_h, img_w = image.shape[:2]

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    return image[y1:y2, x1:x2]


def process_frame(
    frame_path: Path,
    output_path: Path,
    detector: cv2.CascadeClassifier,
    margin: float,
    overwrite: bool,
) -> str:
    """
    Detecta, recorta y guarda la cara de un fotograma.
    Retorna 'saved', 'skipped_exists', o 'skipped_no_face'.
    """
    if output_path.exists() and not overwrite:
        return "skipped_exists"

    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"  [error] No se pudo leer: {frame_path.name}")
        return "skipped_no_face"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return "skipped_no_face"

    # Elegir la cara de mayor área cuando hay varias detectadas
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    cropped = crop_with_margin(image, x, y, w, h, margin)
    resized = cv2.resize(cropped, (IMAGE_WIDTH, IMAGE_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized)

    return "saved"


def process_dataset(
    frame_entries: list[tuple[str, str, Path]],
    detector: cv2.CascadeClassifier,
    margin: float,
    overwrite: bool,
) -> dict:
    stats = {
        "people": set(),
        "views_inspected": set(),
        "frames_processed": 0,
        "faces_saved": 0,
        "skipped_no_face": 0,
        "skipped_exists": 0,
    }

    for person_id, view_name, frame_path in frame_entries:
        stats["people"].add(person_id)
        stats["views_inspected"].add(f"{person_id}/{view_name}")
        stats["frames_processed"] += 1

        faces_dir = frame_path.parent.parent / "faces"
        output_path = faces_dir / frame_path.name

        result = process_frame(frame_path, output_path, detector, margin, overwrite)

        if result == "saved":
            stats["faces_saved"] += 1
        elif result == "skipped_no_face":
            stats["skipped_no_face"] += 1
        elif result == "skipped_exists":
            stats["skipped_exists"] += 1

    return stats


def main() -> None:
    args = parse_args()

    if not PROCESSED_DATASET_DIR.exists():
        print(f"Directorio de datos procesados no encontrado: {PROCESSED_DATASET_DIR}")
        print("Ejecuta primero: python -m src.preprocessing.extract_frames")
        sys.exit(0)

    frame_entries = find_frame_files(PROCESSED_DATASET_DIR)

    if not frame_entries:
        print("No se encontraron fotogramas en data/processed. Ejecuta primero:")
        print("  python -m src.preprocessing.extract_frames")
        sys.exit(0)

    people = {person_id for person_id, _, _ in frame_entries}
    print(f"Personas encontradas  : {len(people)}")
    print(f"Fotogramas encontrados: {len(frame_entries)}")
    print(f"Margen de recorte     : {args.margin}")
    print(f"Sobreescribir         : {'sí' if args.overwrite else 'no'}")
    print()

    detector = get_face_detector()
    stats = process_dataset(frame_entries, detector, args.margin, args.overwrite)

    print()
    print("=== Resumen ===")
    print(f"  Personas             : {len(stats['people'])}")
    print(f"  Vistas inspeccionadas: {len(stats['views_inspected'])}")
    print(f"  Fotogramas procesados: {stats['frames_processed']}")
    print(f"  Caras guardadas      : {stats['faces_saved']}")
    print(f"  Sin cara detectada   : {stats['skipped_no_face']}")
    print(f"  Ya existían          : {stats['skipped_exists']}")


if __name__ == "__main__":
    main()
