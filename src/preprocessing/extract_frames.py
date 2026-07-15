"""
Extracción de fotogramas desde videos crudos.

Input:  data/raw/<person_id>/<view_name>/<video_file>
Output: data/processed/<person_id>/<view_name>/frames/frame_XXXXXX.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

from src.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    SUPPORTED_FACE_VIEWS,
)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_FRAME_INTERVAL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraer fotogramas de vídeos sin procesar a datos/procesados."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_FRAME_INTERVAL,
        help=f"Guardar 1 fotograma cada N fotogramas (default: {DEFAULT_FRAME_INTERVAL})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobreescribir fotogramas existentes.",
    )
    return parser.parse_args()


def find_video_files(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Recorre raw_dir buscando videos con estructura <person_id>/<view_name>/<file>.
    Retorna lista de (person_id, view_name, video_path).
    """
    entries: list[tuple[str, str, Path]] = []

    for person_dir in sorted(raw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for view_dir in sorted(person_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            if view_dir.name not in SUPPORTED_FACE_VIEWS:
                print(f"  [skip] Vista no reconocida: {view_dir.relative_to(raw_dir)}")
                continue
            for video_file in sorted(view_dir.iterdir()):
                if video_file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    entries.append((person_dir.name, view_dir.name, video_file))

    return entries


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    interval: int,
    overwrite: bool,
) -> tuple[int, int]:
    """
    Extrae fotogramas de un video y los guarda como JPG.
    Retorna (frames_saved, frames_skipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] No se pudo abrir: {video_path.name}")
        return 0, 0

    saved = 0
    skipped = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval == 0:
            frame_number = frame_index // interval + 1
            output_path = output_dir / f"frame_{frame_number:06d}.jpg"

            if output_path.exists() and not overwrite:
                skipped += 1
            else:
                cv2.imwrite(str(output_path), frame)
                saved += 1

        frame_index += 1

    cap.release()
    return saved, skipped


def main() -> None:
    args = parse_args()

    if not RAW_DATASET_DIR.exists():
        print(f"Directorio de datos crudos no encontrado: {RAW_DATASET_DIR}")
        sys.exit(0)

    video_entries = find_video_files(RAW_DATASET_DIR)

    if not video_entries:
        print("No se encontraron videos en data/raw. Agrega videos con la estructura:")
        print("  data/raw/<person_id>/<view_name>/<video_file>")
        sys.exit(0)

    people = {person_id for person_id, _, _ in video_entries}
    print(f"Personas encontradas : {len(people)}")
    print(f"Videos encontrados   : {len(video_entries)}")
    print(f"Intervalo de frames  : cada {args.interval} fotogramas")
    print(f"Sobreescribir        : {'sí' if args.overwrite else 'no'}")
    print()

    total_saved = 0
    total_skipped = 0
    videos_processed = 0

    for person_id, view_name, video_path in video_entries:
        output_dir = (
            PROCESSED_DATASET_DIR / person_id / view_name / "frames"
        )
        print(f"  {person_id}/{view_name}/{video_path.name}")

        saved, skipped = extract_frames_from_video(
            video_path, output_dir, args.interval, args.overwrite
        )

        print(f"    -> guardados: {saved}  |  omitidos: {skipped}")
        total_saved += saved
        total_skipped += skipped
        videos_processed += 1

    print()
    print("=== Resumen ===")
    print(f"  Personas        : {len(people)}")
    print(f"  Videos          : {videos_processed}")
    print(f"  Frames guardados: {total_saved}")
    print(f"  Frames omitidos : {total_skipped}")


if __name__ == "__main__":
    main()
