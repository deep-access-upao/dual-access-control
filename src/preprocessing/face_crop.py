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
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MARGIN = 0.20

# Área mínima de la detección como fracción del frame (filtra falsos positivos pequeños)
MIN_FACE_AREA_RATIO = 0.01
MIN_FACE_SIZE = (60, 60)
FRONTAL_MIN_NEIGHBORS = 5
PROFILE_MIN_NEIGHBORS = 4


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


def get_face_detectors() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

    frontal = cv2.CascadeClassifier(frontal_path)
    profile = cv2.CascadeClassifier(profile_path)

    if frontal.empty():
        raise RuntimeError(f"No se pudo cargar el clasificador frontal: {frontal_path}")
    if profile.empty():
        raise RuntimeError(f"No se pudo cargar el clasificador de perfil: {profile_path}")

    return frontal, profile


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


def detect_largest_face(
    gray: np.ndarray,
    detector: cv2.CascadeClassifier,
    min_neighbors: int,
    flip: bool = False,
) -> tuple[int, int, int, int] | None:
    """
    Detecta la cara de mayor área en la imagen en escala de grises.
    Si flip=True, espeja la imagen antes de detectar y corrige las coordenadas al espacio original.
    Descarta detecciones cuya área sea menor que MIN_FACE_AREA_RATIO del frame.
    Retorna (x, y, w, h) o None si no hay cara válida.
    """
    img_h, img_w = gray.shape[:2]
    min_area = img_h * img_w * MIN_FACE_AREA_RATIO

    source = cv2.flip(gray, 1) if flip else gray

    faces = detector.detectMultiScale(
        source, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=MIN_FACE_SIZE
    )

    if len(faces) == 0:
        return None

    valid = [(x, y, w, h) for x, y, w, h in faces if w * h >= min_area]
    if not valid:
        return None

    x, y, w, h = max(valid, key=lambda f: f[2] * f[3])

    if flip:
        x = img_w - x - w

    return x, y, w, h


def find_face_for_view(
    gray: np.ndarray,
    view_name: str,
    frontal_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    """
    Selecciona el cascade adecuado según la vista y devuelve el bounding box de la cara principal.
    Para vistas de perfil, intenta el cascade frontal como respaldo.
    """
    if view_name in ("frontal", "mixed"):
        return detect_largest_face(gray, frontal_detector, FRONTAL_MIN_NEIGHBORS)

    if view_name == "right":
        # Perfil derecho: la cara mira hacia la izquierda en la imagen
        result = detect_largest_face(gray, profile_detector, PROFILE_MIN_NEIGHBORS)
        if result is None:
            result = detect_largest_face(gray, frontal_detector, FRONTAL_MIN_NEIGHBORS)
        return result

    if view_name == "left":
        # Perfil izquierdo: la cara mira hacia la derecha — se detecta espejando
        result = detect_largest_face(gray, profile_detector, PROFILE_MIN_NEIGHBORS, flip=True)
        if result is None:
            result = detect_largest_face(gray, frontal_detector, FRONTAL_MIN_NEIGHBORS)
        return result

    return detect_largest_face(gray, frontal_detector, FRONTAL_MIN_NEIGHBORS)


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
    view_name: str,
    frontal_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
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
    face = find_face_for_view(gray, view_name, frontal_detector, profile_detector)

    if face is None:
        return "skipped_no_face"

    x, y, w, h = face
    cropped = crop_with_margin(image, x, y, w, h, margin)
    resized = cv2.resize(cropped, (IMAGE_WIDTH, IMAGE_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized)

    return "saved"


def process_dataset(
    frame_entries: list[tuple[str, str, Path]],
    frontal_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
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

        result = process_frame(
            frame_path, output_path, view_name,
            frontal_detector, profile_detector,
            margin, overwrite,
        )

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
    print(f"Sobreescribir         : {'si' if args.overwrite else 'no'}")
    print()

    frontal_detector, profile_detector = get_face_detectors()
    stats = process_dataset(frame_entries, frontal_detector, profile_detector, args.margin, args.overwrite)

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
