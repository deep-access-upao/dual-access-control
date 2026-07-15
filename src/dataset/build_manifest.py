"""Construye el manifiesto reproducible de imágenes faciales procesadas.

El manifiesto conserva una fila por archivo, incluido ``rejected/``. Las rutas
son relativas al repositorio para que el resultado sea portable. En el layout
legado los frames no guardan el nombre del video: cuando existe un único video
crudo para persona/vista se recupera ese origen; si hay varios, se usa un grupo
conservador que mantiene junta toda la vista y evita fuga entre splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
from pathlib import Path

from src.config import (
    DATASET_MANIFEST_PATH,
    PROCESSED_DATASET_DIR,
    PROJECT_ROOT,
    RAW_DATASET_DIR,
    SUPPORTED_FACE_VIEWS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
QUALITY_REPORT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "face_quality_report.csv"

MANIFEST_COLUMNS = [
    "image_path",
    "person_id",
    "source_video",
    "source_mapping",
    "view",
    "status",
    "quality_status",
    "quality_reasons",
    "width",
    "height",
    "channels",
    "file_size_bytes",
    "sha256",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera el manifiesto del dataset facial.")
    parser.add_argument("--output", type=Path, default=DATASET_MANIFEST_PATH)
    return parser.parse_args()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_quality_report(report_path: Path = QUALITY_REPORT_PATH) -> dict[str, dict[str, str]]:
    """Indexa el reporte previo por ruta absoluta normalizada y por ruta relativa."""
    if not report_path.is_file():
        return {}

    records: dict[str, dict[str, str]] = {}
    with report_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            raw_path = row.get("path", "")
            if not raw_path:
                continue
            path = Path(raw_path)
            absolute = path if path.is_absolute() else PROJECT_ROOT / path
            records[str(absolute.resolve()).casefold()] = row
            try:
                records[relative_path(absolute).casefold()] = row
            except ValueError:
                pass
    return records


def find_raw_videos(person_id: str, view: str) -> list[Path]:
    directory = RAW_DATASET_DIR / person_id / view
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def infer_source_video(person_id: str, view: str) -> tuple[str, str]:
    videos = find_raw_videos(person_id, view)
    if len(videos) == 1:
        return relative_path(videos[0]), "unique_raw_video"

    # Agrupar toda la vista es deliberadamente conservador: aunque el origen
    # exacto sea ambiguo, ningún frame potencialmente consecutivo cruza splits.
    group = f"legacy_group/{person_id}/{view}"
    mapping = "missing_raw_video" if not videos else "ambiguous_legacy_group"
    return group, mapping


def iter_processed_images(processed_dir: Path = PROCESSED_DATASET_DIR):
    if not processed_dir.is_dir():
        return
    for person_dir in sorted(processed_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for view_dir in sorted(person_dir.iterdir()):
            if not view_dir.is_dir() or view_dir.name not in SUPPORTED_FACE_VIEWS:
                continue
            state_directories = (
                ("faces", view_dir / "faces"),
                ("rejected", view_dir / "rejected"),
                # Algunas limpiezas previas usaron faces/rejected; se conserva
                # compatibilidad para no ocultar archivos rechazados existentes.
                ("rejected", view_dir / "faces" / "rejected"),
            )
            for state_dir_name, state_dir in state_directories:
                if not state_dir.is_dir():
                    continue
                for image_path in sorted(state_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        yield person_dir.name, view_dir.name, state_dir_name, image_path


def inspect_image(path: Path) -> tuple[int | str, int | str, int | str]:
    """Lee dimensiones PNG/JPEG sin cargar librerías de visión ni decodificar píxeles."""
    try:
        with path.open("rb") as file:
            signature = file.read(24)
            if signature.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", signature[16:24])
                return width, height, 3
            if signature[:2] != b"\xff\xd8":
                return "", "", ""

            file.seek(2)
            while True:
                marker_start = file.read(1)
                if not marker_start:
                    return "", "", ""
                if marker_start != b"\xff":
                    continue
                marker = file.read(1)
                while marker == b"\xff":
                    marker = file.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                length_bytes = file.read(2)
                if len(length_bytes) != 2:
                    return "", "", ""
                segment_length = struct.unpack(">H", length_bytes)[0]
                sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                               0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
                if marker and marker[0] in sof_markers:
                    data = file.read(6)
                    if len(data) != 6:
                        return "", "", ""
                    height, width, channels = struct.unpack(">xHHB", data)
                    return width, height, channels
                file.seek(segment_length - 2, 1)
    except (OSError, struct.error):
        return "", "", ""


def build_manifest() -> list[dict[str, object]]:
    quality = load_quality_report()
    rows: list[dict[str, object]] = []
    source_cache: dict[tuple[str, str], tuple[str, str]] = {}

    for person_id, view, state_dir, image_path in iter_processed_images():
        key = (person_id, view)
        source_video, source_mapping = source_cache.setdefault(
            key, infer_source_video(person_id, view)
        )
        rel_path = relative_path(image_path)
        quality_row = quality.get(str(image_path.resolve()).casefold(), quality.get(rel_path.casefold(), {}))
        quality_status = quality_row.get("status", "not_evaluated")
        quality_reasons = quality_row.get("reasons", "")
        width, height, channels = inspect_image(image_path)

        if state_dir == "rejected" or quality_status in {"review", "rejected"}:
            status = "rejected"
        elif width == "":
            status = "rejected"
            quality_reasons = quality_reasons or "invalid_image"
        else:
            status = "usable"

        rows.append({
            "image_path": rel_path,
            "person_id": person_id,
            "source_video": source_video,
            "source_mapping": source_mapping,
            "view": view,
            "status": status,
            "quality_status": quality_status,
            "quality_reasons": quality_reasons,
            "width": width,
            "height": height,
            "channels": channels,
            "file_size_bytes": image_path.stat().st_size,
            "sha256": file_sha256(image_path),
            "split": "",
        })
    return rows


def write_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_manifest()
    if not rows:
        raise SystemExit("ERROR: no se encontraron imágenes en data/processed.")
    write_manifest(rows, args.output)
    usable = sum(row["status"] == "usable" for row in rows)
    rejected = len(rows) - usable
    videos = len({row["source_video"] for row in rows})
    print(f"Manifiesto: {args.output}")
    print(f"Imágenes: {len(rows)} (usable={usable}, rejected={rejected})")
    print(f"Personas: {len({row['person_id'] for row in rows})}")
    print(f"Videos/grupos de origen: {videos}")


if __name__ == "__main__":
    main()
