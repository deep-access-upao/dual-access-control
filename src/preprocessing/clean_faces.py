"""
Inspección de calidad de imágenes de rostros recortados.

Input:  data/processed/<person_id>/<view_name>/faces/<face_file>.jpg
Output: outputs/metrics/face_quality_report.csv
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    METRICS_DIR,
    PROCESSED_DATASET_DIR,
    SUPPORTED_FACE_VIEWS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REPORT_PATH = METRICS_DIR / "face_quality_report.csv"

CSV_COLUMNS = [
    "path",
    "person_id",
    "view",
    "width",
    "height",
    "blur_score",
    "brightness",
    "contrast",
    "face_detected_again",
    "status",
    "reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspeccionar calidad de imágenes de rostros recortados."
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=30.0,
        help="Umbral mínimo de nitidez (varianza del Laplaciano). Default: 30.0",
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=35.0,
        help="Brillo mínimo aceptable (media de píxeles en escala de grises). Default: 35.0",
    )
    parser.add_argument(
        "--max-brightness",
        type=float,
        default=220.0,
        help="Brillo máximo aceptable (media de píxeles en escala de grises). Default: 220.0",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=15.0,
        help="Contraste mínimo aceptable (desviación estándar en escala de grises). Default: 15.0",
    )
    parser.add_argument(
        "--move-review",
        action="store_true",
        help="Mover imágenes marcadas como 'review' a <view>/rejected/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobreescribir archivos en el destino al mover imágenes.",
    )
    return parser.parse_args()


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar el clasificador Haar: {cascade_path}")
    return detector


def find_face_files(processed_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Busca archivos de cara con estructura <person_id>/<view_name>/faces/<file>.
    Retorna lista de (person_id, view_name, face_path).
    """
    entries: list[tuple[str, str, Path]] = []

    for person_dir in sorted(processed_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for view_dir in sorted(person_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            if view_dir.name not in SUPPORTED_FACE_VIEWS:
                continue
            faces_dir = view_dir / "faces"
            if not faces_dir.is_dir():
                continue
            for face_file in sorted(faces_dir.iterdir()):
                if face_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    entries.append((person_dir.name, view_dir.name, face_file))

    return entries


def compute_quality_metrics(
    image: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> dict:
    """
    Calcula métricas de calidad para un recorte facial.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # minNeighbors menor para ser más sensible dentro del recorte ya ajustado
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    face_detected_again = len(faces) > 0

    return {
        "blur_score": round(blur_score, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "face_detected_again": face_detected_again,
    }


def inspect_image(
    face_path: Path,
    person_id: str,
    view: str,
    detector: cv2.CascadeClassifier,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
    min_contrast: float,
) -> dict:
    """
    Inspecciona una imagen de rostro y devuelve su fila de reporte.
    """
    record: dict = {
        "path": str(face_path),
        "person_id": person_id,
        "view": view,
        "width": "",
        "height": "",
        "blur_score": "",
        "brightness": "",
        "contrast": "",
        "face_detected_again": "",
        "status": "ok",
        "reasons": "",
    }

    image = cv2.imread(str(face_path))

    if image is None:
        record["status"] = "review"
        record["reasons"] = "invalid_image"
        return record

    h, w = image.shape[:2]
    record["width"] = w
    record["height"] = h

    reasons: list[str] = []

    if w != IMAGE_WIDTH or h != IMAGE_HEIGHT:
        reasons.append("unexpected_size")

    metrics = compute_quality_metrics(image, detector)
    record["blur_score"] = metrics["blur_score"]
    record["brightness"] = metrics["brightness"]
    record["contrast"] = metrics["contrast"]
    record["face_detected_again"] = metrics["face_detected_again"]

    if metrics["blur_score"] < blur_threshold:
        reasons.append("blurry")
    if metrics["brightness"] < min_brightness:
        reasons.append("too_dark")
    if metrics["brightness"] > max_brightness:
        reasons.append("too_bright")
    if metrics["contrast"] < min_contrast:
        reasons.append("low_contrast")
    if not metrics["face_detected_again"]:
        reasons.append("no_face_detected_in_crop")

    if reasons:
        record["status"] = "review"
        record["reasons"] = "|".join(reasons)

    return record


def move_to_rejected(face_path: Path, overwrite: bool) -> bool:
    """
    Mueve una imagen de faces/ a rejected/ dentro del mismo directorio de vista.
    Retorna True si el archivo fue movido efectivamente.
    """
    rejected_dir = face_path.parent.parent / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    dest_path = rejected_dir / face_path.name

    if dest_path.exists() and not overwrite:
        return False

    shutil.move(str(face_path), str(dest_path))
    return True


def write_report(records: list[dict], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()

    if not PROCESSED_DATASET_DIR.exists():
        print(f"Directorio de datos procesados no encontrado: {PROCESSED_DATASET_DIR}")
        print("Ejecuta primero: python -m src.preprocessing.face_crop")
        sys.exit(0)

    face_entries = find_face_files(PROCESSED_DATASET_DIR)

    if not face_entries:
        print("No se encontraron imágenes de rostros en data/processed.")
        print("Ejecuta primero: python -m src.preprocessing.face_crop")
        sys.exit(0)

    print(f"Imágenes encontradas  : {len(face_entries)}")
    print(f"Umbral de nitidez     : {args.blur_threshold}")
    print(f"Brillo mínimo/máximo  : {args.min_brightness} / {args.max_brightness}")
    print(f"Contraste mínimo      : {args.min_contrast}")
    print(f"Mover rechazados      : {'sí' if args.move_review else 'no'}")
    print()

    detector = get_face_detector()
    records: list[dict] = []

    for person_id, view_name, face_path in face_entries:
        record = inspect_image(
            face_path=face_path,
            person_id=person_id,
            view=view_name,
            detector=detector,
            blur_threshold=args.blur_threshold,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
            min_contrast=args.min_contrast,
        )
        records.append(record)

    ok_count = sum(1 for r in records if r["status"] == "ok")
    review_count = sum(1 for r in records if r["status"] == "review")
    moved_count = 0

    if args.move_review:
        for record in records:
            if record["status"] == "review":
                face_path = Path(record["path"])
                if face_path.exists():
                    moved = move_to_rejected(face_path, args.overwrite)
                    if moved:
                        moved_count += 1

    write_report(records, REPORT_PATH)

    print("=== Resumen ===")
    print(f"  Total inspeccionadas : {len(records)}")
    print(f"  Estado ok            : {ok_count}")
    print(f"  Estado review        : {review_count}")
    if args.move_review:
        print(f"  Movidas a rejected/  : {moved_count}")
    print(f"  Reporte generado     : {REPORT_PATH}")


if __name__ == "__main__":
    main()
