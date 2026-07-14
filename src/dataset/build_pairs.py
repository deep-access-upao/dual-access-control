"""Genera pares dentro de splits de imágenes ya asignados.

Orden obligatorio del protocolo: build_manifest -> build_splits -> build_pairs.
Nunca se construyen pares entre splits y la clave canónica ignora el orden A/B.
El muestreo reparte positivos entre identidades y negativos entre combinaciones
de identidades para limitar el sesgo de las personas con más imágenes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from collections import defaultdict
from pathlib import Path

from src.config import DATASET_MANIFEST_PATH, PAIRS_DIR
from src.dataset.build_splits import SPLITS, read_manifest

CSV_COLUMNS = [
    "image_a",
    "image_b",
    "label",
    "person_a",
    "person_b",
    "source_video_a",
    "source_video_b",
    "view_a",
    "view_b",
    "split",
]

DEFAULT_PAIR_COUNTS = {"train": 4000, "validation": 500, "test": 500}
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera pares balanceados después del split.")
    parser.add_argument("--manifest", type=Path, default=DATASET_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=PAIRS_DIR)
    parser.add_argument("--train-pairs", type=int, default=DEFAULT_PAIR_COUNTS["train"])
    parser.add_argument("--validation-pairs", type=int, default=DEFAULT_PAIR_COUNTS["validation"])
    parser.add_argument("--test-pairs", type=int, default=DEFAULT_PAIR_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pair_key(image_a: str, image_b: str) -> tuple[str, str]:
    return tuple(sorted((image_a, image_b)))


def allocate_evenly(total: int, keys: list[object]) -> dict[object, int]:
    if not keys:
        return {}
    base, remainder = divmod(total, len(keys))
    return {key: base + (index < remainder) for index, key in enumerate(keys)}


def make_row(a: dict[str, str], b: dict[str, str], label: int, split: str) -> dict[str, object]:
    if a["image_path"] > b["image_path"]:
        a, b = b, a
    return {
        "image_a": a["image_path"],
        "image_b": b["image_path"],
        "label": label,
        "person_a": a["person_id"],
        "person_b": b["person_id"],
        "source_video_a": a["source_video"],
        "source_video_b": b["source_video"],
        "view_a": a["view"],
        "view_b": b["view"],
        "split": split,
    }


def positive_pairs(
    rows: list[dict[str, str]], target: int, split: str, rng: random.Random
) -> list[dict[str, object]]:
    by_person: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_person[row["person_id"]].append(row)

    candidates: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    for person_id, images in sorted(by_person.items()):
        pairs = list(itertools.combinations(sorted(images, key=lambda row: row["image_path"]), 2))
        rng.shuffle(pairs)
        if pairs:
            candidates[person_id] = pairs

    people = list(candidates)
    rng.shuffle(people)
    quotas = allocate_evenly(target, people)
    selected: list[dict[str, object]] = []
    shortage = 0
    for person_id in people:
        take = min(quotas[person_id], len(candidates[person_id]))
        selected.extend(make_row(a, b, 1, split) for a, b in candidates[person_id][:take])
        shortage += quotas[person_id] - take

    if shortage:
        remaining = [pair for person_id in people for pair in candidates[person_id][quotas[person_id]:]]
        rng.shuffle(remaining)
        selected.extend(make_row(a, b, 1, split) for a, b in remaining[:shortage])
    return selected[:target]


def sample_product(
    images_a: list[dict[str, str]],
    images_b: list[dict[str, str]],
    target: int,
    rng: random.Random,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    capacity = len(images_a) * len(images_b)
    if target >= capacity:
        pairs = list(itertools.product(images_a, images_b))
        rng.shuffle(pairs)
        return pairs

    chosen: set[tuple[int, int]] = set()
    while len(chosen) < target:
        chosen.add((rng.randrange(len(images_a)), rng.randrange(len(images_b))))
    return [(images_a[i], images_b[j]) for i, j in chosen]


def negative_pairs(
    rows: list[dict[str, str]], target: int, split: str, rng: random.Random
) -> list[dict[str, object]]:
    by_person: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_person[row["person_id"]].append(row)
    person_pairs = list(itertools.combinations(sorted(by_person), 2))
    rng.shuffle(person_pairs)
    quotas = allocate_evenly(target, person_pairs)

    selected: list[dict[str, object]] = []
    shortage = 0
    for person_a, person_b in person_pairs:
        images_a = by_person[person_a]
        images_b = by_person[person_b]
        quota = quotas[(person_a, person_b)]
        sampled = sample_product(images_a, images_b, min(quota, len(images_a) * len(images_b)), rng)
        selected.extend(make_row(a, b, 0, split) for a, b in sampled)
        shortage += quota - len(sampled)

    if shortage:
        existing = {pair_key(str(row["image_a"]), str(row["image_b"])) for row in selected}
        attempts = 0
        max_attempts = max(1000, shortage * 100)
        people = sorted(by_person)
        while shortage and attempts < max_attempts:
            attempts += 1
            person_a, person_b = rng.sample(people, 2)
            a, b = rng.choice(by_person[person_a]), rng.choice(by_person[person_b])
            key = pair_key(a["image_path"], b["image_path"])
            if key in existing:
                continue
            existing.add(key)
            selected.append(make_row(a, b, 0, split))
            shortage -= 1
    return selected[:target]


def generate_split_pairs(
    rows: list[dict[str, str]], target: int, split: str, seed: int
) -> list[dict[str, object]]:
    if target < 2:
        raise ValueError(f"{split}: se requieren al menos 2 pares.")
    rng = random.Random(f"{seed}:{split}")
    positive_target = target // 2
    negative_target = target - positive_target
    positives = positive_pairs(rows, positive_target, split, rng)
    negatives = negative_pairs(rows, negative_target, split, rng)
    if len(positives) != positive_target or len(negatives) != negative_target:
        raise ValueError(
            f"{split}: combinaciones insuficientes; solicitadas {positive_target}/{negative_target} "
            f"positivas/negativas, disponibles {len(positives)}/{len(negatives)}."
        )
    result = positives + negatives
    rng.shuffle(result)
    return result


def output_path(output_dir: Path, split: str) -> Path:
    # Se conserva val_pairs.csv por compatibilidad con entrenamiento/notebooks
    # existentes; la columna interna usa el nombre explícito "validation".
    filename_split = "val" if split == "validation" else split
    return output_dir / f"{filename_split}_pairs.csv"


def write_pairs(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    counts = {
        "train": args.train_pairs,
        "validation": args.validation_pairs,
        "test": args.test_pairs,
    }
    paths = [output_path(args.output_dir, split) for split in SPLITS]
    if not args.overwrite and any(path.exists() for path in paths):
        existing = ", ".join(str(path) for path in paths if path.exists())
        raise SystemExit(f"ERROR: ya existen CSV ({existing}). Usa --overwrite.")

    manifest = read_manifest(args.manifest)
    usable = [row for row in manifest if row["status"] == "usable"]
    if any(not row["split"] for row in usable):
        raise SystemExit("ERROR: hay imágenes usables sin split. Ejecuta primero build_splits.")

    all_keys: set[tuple[str, str]] = set()
    for split in SPLITS:
        split_images = [row for row in usable if row["split"] == split]
        pairs = generate_split_pairs(split_images, counts[split], split, args.seed)
        keys = {pair_key(str(row["image_a"]), str(row["image_b"])) for row in pairs}
        if len(keys) != len(pairs) or all_keys.intersection(keys):
            raise RuntimeError("Se detectaron pares duplicados durante la generación.")
        all_keys.update(keys)
        path = output_path(args.output_dir, split)
        write_pairs(pairs, path)
        positives = sum(int(row["label"]) == 1 for row in pairs)
        print(f"{split:10s}: {len(pairs):4d} pares ({positives} positivos, {len(pairs)-positives} negativos) -> {path}")


if __name__ == "__main__":
    main()
