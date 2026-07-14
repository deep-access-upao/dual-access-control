"""Audita manifiesto, splits y pares; retorna código 1 ante cualquier fuga."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import DATASET_MANIFEST_PATH, PAIRS_DIR
from src.dataset.build_pairs import CSV_COLUMNS, output_path, pair_key
from src.dataset.build_splits import SPLITS, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida que splits y pares no tengan fuga.")
    parser.add_argument("--manifest", type=Path, default=DATASET_MANIFEST_PATH)
    parser.add_argument("--pairs-dir", type=Path, default=PAIRS_DIR)
    return parser.parse_args()


def load_pairs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Falta el CSV de pares: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    missing = set(CSV_COLUMNS) - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")
    return rows


def intersection_count(values_by_split: dict[str, set[str]]) -> int:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, values in values_by_split.items():
        for value in values:
            owners[value].add(split)
    return sum(len(splits) > 1 for splits in owners.values())


def audit(manifest_path: Path, pairs_dir: Path) -> tuple[dict[str, object], list[str]]:
    manifest = read_manifest(manifest_path)
    usable = [row for row in manifest if row["status"] == "usable"]
    errors: list[str] = []

    manifest_path_counts = Counter(row["image_path"] for row in manifest)
    repeated_manifest_paths = sum(count > 1 for count in manifest_path_counts.values())
    if repeated_manifest_paths:
        errors.append(f"{repeated_manifest_paths} rutas repetidas en el manifiesto")

    unassigned = [row for row in usable if row["split"] not in SPLITS]
    if unassigned:
        errors.append(f"{len(unassigned)} imágenes usables sin split válido")

    image_sets = {split: {row["image_path"] for row in usable if row["split"] == split} for split in SPLITS}
    video_sets = {split: {row["source_video"] for row in usable if row["split"] == split} for split in SPLITS}
    image_overlap = intersection_count(image_sets)
    video_overlap = intersection_count(video_sets)
    if image_overlap:
        errors.append(f"{image_overlap} imágenes compartidas entre splits")
    if video_overlap:
        errors.append(f"{video_overlap} videos compartidos entre splits")

    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in usable:
        hashes[row["sha256"]].append(row)
    duplicate_hash_groups = sum(len(rows) > 1 for rows in hashes.values())
    cross_split_hashes = sum(
        len({row["split"] for row in rows}) > 1 for rows in hashes.values()
    )
    if cross_split_hashes:
        errors.append(f"{cross_split_hashes} hashes exactos compartidos entre splits")

    image_index = {row["image_path"]: row for row in usable}
    pairs_by_split = {split: load_pairs(output_path(pairs_dir, split)) for split in SPLITS}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    pair_balance: dict[str, dict[str, int]] = {}

    for split, pairs in pairs_by_split.items():
        keys: list[tuple[str, str]] = []
        positives = negatives = 0
        for line_number, pair in enumerate(pairs, start=2):
            key = pair_key(pair["image_a"], pair["image_b"])
            keys.append(key)
            a = image_index.get(pair["image_a"])
            b = image_index.get(pair["image_b"])
            if a is None or b is None:
                errors.append(f"{split}:{line_number}: imagen ausente/no usable en manifiesto")
                continue
            if pair["split"] != split or a["split"] != split or b["split"] != split:
                errors.append(f"{split}:{line_number}: par con imágenes de otro split")
            expected_metadata = {
                "person_a": a["person_id"],
                "person_b": b["person_id"],
                "source_video_a": a["source_video"],
                "source_video_b": b["source_video"],
                "view_a": a["view"],
                "view_b": b["view"],
            }
            if any(pair[column] != value for column, value in expected_metadata.items()):
                errors.append(f"{split}:{line_number}: metadatos del par inconsistentes")
            expected_label = "1" if a["person_id"] == b["person_id"] else "0"
            if pair["label"] != expected_label:
                errors.append(f"{split}:{line_number}: etiqueta inconsistente")
            positives += pair["label"] == "1"
            negatives += pair["label"] == "0"
        duplicates = len(keys) - len(set(keys))
        if duplicates:
            errors.append(f"{split}: {duplicates} pares repetidos ignorando orden A/B")
        pair_sets[split] = set(keys)
        pair_balance[split] = {"positive": positives, "negative": negatives, "total": len(pairs)}

    repeated_pairs_across_splits = intersection_count(pair_sets)
    if repeated_pairs_across_splits:
        errors.append(f"{repeated_pairs_across_splits} pares repetidos entre splits")

    split_summary: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        rows = [row for row in usable if row["split"] == split]
        split_summary[split] = {
            "images": len(rows),
            "videos": len({row["source_video"] for row in rows}),
            "people": len({row["person_id"] for row in rows}),
            "images_by_person": dict(sorted(Counter(row["person_id"] for row in rows).items())),
            **pair_balance[split],
        }

    report: dict[str, object] = {
        "shared_images": image_overlap,
        "shared_videos": video_overlap,
        "repeated_pairs_across_splits": repeated_pairs_across_splits,
        "duplicate_hash_groups": duplicate_hash_groups,
        "cross_split_hashes": cross_split_hashes,
        "splits": split_summary,
    }
    return report, errors


def print_report(report: dict[str, object], errors: list[str]) -> None:
    print("=== Auditoría de fuga ===")
    print(f"Imágenes compartidas entre splits : {report['shared_images']}")
    print(f"Videos compartidos entre splits   : {report['shared_videos']}")
    print(f"Pares repetidos entre splits      : {report['repeated_pairs_across_splits']}")
    print(f"Grupos de hash duplicado          : {report['duplicate_hash_groups']}")
    print(f"Hashes duplicados entre splits    : {report['cross_split_hashes']}")
    print()
    for split, summary in report["splits"].items():
        print(
            f"{split:10s}: imágenes={summary['images']:4d}, videos={summary['videos']:2d}, "
            f"personas={summary['people']:2d}, pares={summary['total']:4d} "
            f"(positivos={summary['positive']}, negativos={summary['negative']})"
        )
        distribution = ", ".join(f"{person}={count}" for person, count in summary["images_by_person"].items())
        print(f"  imágenes por persona: {distribution}")
    print()
    if errors:
        print("AUDITORÍA FALLIDA")
        for error in errors:
            print(f"  - {error}")
    else:
        print("AUDITORÍA APROBADA: no se detectó fuga entre splits.")


def main() -> None:
    args = parse_args()
    try:
        report, errors = audit(args.manifest, args.pairs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"AUDITORÍA FALLIDA: {exc}")
        raise SystemExit(1) from exc
    print_report(report, errors)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
