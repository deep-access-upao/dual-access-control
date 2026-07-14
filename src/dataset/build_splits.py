"""Asigna splits operacionales a imágenes antes de generar pares.

La unidad indivisible es ``source_video``. La asignación es estratificada por
persona para mantener identidades conocidas en train/validation/test, sin
compartir capturas. Con cuatro videos por persona, la configuración por defecto
produce 2/1/1 videos (50/25/25), apropiada para este dataset pequeño.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.config import DATASET_MANIFEST_PATH
from src.dataset.build_manifest import MANIFEST_COLUMNS

SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.50, 0.25, 0.25)
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asigna splits sin fuga por video fuente.")
    parser.add_argument("--manifest", type=Path, default=DATASET_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero build_manifest.")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    missing = set(MANIFEST_COLUMNS) - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"El manifiesto no contiene columnas requeridas: {sorted(missing)}")
    return rows


def split_counts(group_count: int, ratios: tuple[float, float, float]) -> list[int]:
    if group_count < 3:
        raise ValueError(
            "El protocolo operacional necesita al menos 3 videos/grupos por persona "
            "para representar train, validation y test sin compartir videos."
        )
    raw = [group_count * ratio for ratio in ratios]
    counts = [int(value) for value in raw]

    # Garantiza presencia en los tres splits antes de distribuir el remanente.
    counts = [max(1, count) for count in counts]
    while sum(counts) > group_count:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("No se pueden satisfacer los ratios con estos grupos.")
        index = min(candidates, key=lambda i: raw[i] - counts[i])
        counts[index] -= 1
    while sum(counts) < group_count:
        index = max(range(3), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    return counts


def assign_splits(
    rows: list[dict[str, str]],
    seed: int,
    ratios: tuple[float, float, float],
) -> list[dict[str, str]]:
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Los ratios deben ser positivos y sumar 1.0.")

    usable = [row for row in rows if row["status"] == "usable"]
    duplicate_origins: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in usable:
        duplicate_origins[row["sha256"]].add((row["person_id"], row["source_video"]))
    ambiguous_hashes = {digest: origins for digest, origins in duplicate_origins.items() if len(origins) > 1}
    if ambiguous_hashes:
        raise ValueError(
            f"Hay {len(ambiguous_hashes)} hash(es) presentes en distintos videos/personas. "
            "Deduplica o corrige el origen antes de asignar splits."
        )

    by_person: dict[str, set[str]] = defaultdict(set)
    for row in usable:
        by_person[row["person_id"]].add(row["source_video"])

    assignments: dict[str, str] = {}
    for person_id in sorted(by_person):
        videos = sorted(by_person[person_id])
        rng = random.Random(f"{seed}:{person_id}")
        rng.shuffle(videos)
        counts = split_counts(len(videos), ratios)
        offset = 0
        for split, count in zip(SPLITS, counts):
            for video in videos[offset:offset + count]:
                assignments[video] = split
            offset += count

    result: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        updated["split"] = assignments.get(row["source_video"], "") if row["status"] == "usable" else ""
        result.append(updated)
    return result


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    assigned = assign_splits(rows, args.seed, ratios)
    write_manifest(assigned, args.manifest)

    print(f"Splits asignados en: {args.manifest}")
    for split in SPLITS:
        subset = [row for row in assigned if row["split"] == split]
        print(
            f"{split:10s}: imágenes={len(subset):4d}, "
            f"videos={len({row['source_video'] for row in subset}):2d}, "
            f"personas={len({row['person_id'] for row in subset}):2d}"
        )


if __name__ == "__main__":
    main()
