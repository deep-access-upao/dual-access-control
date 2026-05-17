"""
Genera pares positivos y negativos de imágenes para el entrenamiento de la red Siamesa.

Lee imágenes de data/processed y escribe los CSV en data/pairs:
    data/pairs/train_pairs.csv
    data/pairs/val_pairs.csv
    data/pairs/test_pairs.csv
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

from src.config import (
    PAIRS_DIR,
    PROCESSED_DATASET_DIR,
    PROJECT_ROOT,
    SUPPORTED_FACE_VIEWS,
)

FACE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CSV_COLUMNS = [
    "image_a",
    "image_b",
    "label",
    "person_a",
    "person_b",
    "view_a",
    "view_b",
    "split",
]

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test recibe el resto: 1.0 - TRAIN_RATIO - VAL_RATIO

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera pares positivos y negativos para el entrenamiento de la red Siamesa."
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        metavar="N",
        help="Número máximo total de pares (positivos + negativos). Por defecto: ilimitado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Semilla aleatoria para reproducibilidad. Por defecto: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobreescribir archivos CSV existentes en data/pairs.",
    )
    return parser.parse_args()


def collect_face_images(
    processed_dir: Path,
    supported_views: list[str],
) -> dict[str, dict[str, list[Path]]]:
    """
    Retorna {person_id: {view_name: [absolute_path, ...]}}
    Solo incluye vistas con al menos una imagen de cara presente.
    """
    people: dict[str, dict[str, list[Path]]] = {}

    if not processed_dir.exists():
        return people

    for person_dir in sorted(processed_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        views: dict[str, list[Path]] = {}

        for view_dir in sorted(person_dir.iterdir()):
            if not view_dir.is_dir() or view_dir.name not in supported_views:
                continue

            faces_dir = view_dir / "faces"
            if not faces_dir.is_dir():
                continue

            images = [
                f
                for f in sorted(faces_dir.iterdir())
                if f.is_file() and f.suffix.lower() in FACE_EXTENSIONS
            ]

            if images:
                views[view_dir.name] = images

        if views:
            people[person_dir.name] = views

    return people


def generate_positive_pairs(
    people: dict[str, dict[str, list[Path]]],
    rng: random.Random,
) -> list[tuple]:
    """
    Retorna todos los pares únicos de la misma persona entre todas las vistas disponibles.
    Incluye pares de la misma vista y de vistas distintas (frontal-left, etc.).
    """
    pairs: list[tuple] = []

    for person_id, views in people.items():
        # Aplanar imágenes de la persona junto con su vista de origen
        all_images: list[tuple[Path, str]] = [
            (img, view_name)
            for view_name, images in views.items()
            for img in images
        ]

        for (img_a, view_a), (img_b, view_b) in itertools.combinations(all_images, 2):
            pairs.append((img_a, img_b, person_id, person_id, view_a, view_b))

    rng.shuffle(pairs)
    return pairs


def generate_negative_pairs(
    people: dict[str, dict[str, list[Path]]],
    n_pairs: int,
    rng: random.Random,
) -> list[tuple]:
    """
    Retorna hasta n_pairs de personas distintas usando muestreo aleatorio.
    Usa un conjunto seen para evitar duplicados exactos.
    """
    person_ids = list(people.keys())
    pairs: list[tuple] = []
    seen: set[tuple[str, str]] = set()

    # Límite de intentos para datasets pequeños con pocas combinaciones únicas
    max_attempts = n_pairs * 20
    attempts = 0

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1

        person_a, person_b = rng.sample(person_ids, 2)
        view_a = rng.choice(list(people[person_a].keys()))
        view_b = rng.choice(list(people[person_b].keys()))
        img_a = rng.choice(people[person_a][view_a])
        img_b = rng.choice(people[person_b][view_b])

        key = (str(img_a), str(img_b))
        if key in seen:
            continue

        seen.add(key)
        pairs.append((img_a, img_b, person_a, person_b, view_a, view_b))

    return pairs


def split_pairs(
    pairs: list[tuple],
    rng: random.Random,
) -> list[tuple]:
    """
    Mezcla los pares y asigna la etiqueta de split ('train', 'val', 'test').
    Retorna una lista de tuplas con el split añadido al final.
    """
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    labeled: list[tuple] = []
    for i, pair in enumerate(shuffled):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        labeled.append(pair + (split,))

    return labeled


def write_pairs_csv(
    labeled_pairs: list[tuple],
    pairs_dir: Path,
    project_root: Path,
) -> None:
    """
    Escribe train_pairs.csv, val_pairs.csv y test_pairs.csv en pairs_dir.
    Las rutas de imagen son relativas a project_root.
    """
    pairs_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for img_a, img_b, person_a, person_b, view_a, view_b, split in labeled_pairs:
        label = 1 if person_a == person_b else 0
        row = {
            "image_a": img_a.relative_to(project_root).as_posix(),
            "image_b": img_b.relative_to(project_root).as_posix(),
            "label": label,
            "person_a": person_a,
            "person_b": person_b,
            "view_a": view_a,
            "view_b": view_b,
            "split": split,
        }
        by_split[split].append(row)

    for split_name, rows in by_split.items():
        csv_path = pairs_dir / f"{split_name}_pairs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {len(rows):>5} pares → {csv_path.relative_to(project_root).as_posix()}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Verificar si los CSV ya existen cuando --overwrite no fue indicado
    if not args.overwrite:
        existing = [
            p
            for p in [
                PAIRS_DIR / "train_pairs.csv",
                PAIRS_DIR / "val_pairs.csv",
                PAIRS_DIR / "test_pairs.csv",
            ]
            if p.exists()
        ]
        if existing:
            print("Los archivos CSV ya existen. Usa --overwrite para reemplazarlos:")
            for p in existing:
                print(f"  {p.relative_to(PROJECT_ROOT).as_posix()}")
            return

    print(f"Analizando: {PROCESSED_DATASET_DIR.relative_to(PROJECT_ROOT).as_posix()}")
    people = collect_face_images(PROCESSED_DATASET_DIR, SUPPORTED_FACE_VIEWS)

    if len(people) < 2:
        print(
            f"No hay suficientes personas en los datos procesados "
            f"(encontradas: {len(people)}, se necesitan al menos 2). "
            f"Ejecuta el preprocesamiento primero."
        )
        return

    total_images = sum(len(imgs) for views in people.values() for imgs in views.values())
    print(f"Personas encontradas: {len(people)}, imágenes de cara: {total_images}.")

    positive_pairs = generate_positive_pairs(people, rng)

    if not positive_pairs:
        print("No se generaron pares positivos. Cada persona necesita al menos 2 imágenes de cara.")
        return

    # Determinar cuántos pares generar por clase para mantener el balance
    if args.max_pairs is not None:
        target_per_class = max(1, args.max_pairs // 2)
    else:
        target_per_class = len(positive_pairs)

    positive_pairs = positive_pairs[:target_per_class]
    negative_pairs = generate_negative_pairs(people, target_per_class, rng)

    if not negative_pairs:
        print("No se pudieron generar pares negativos. Se necesitan al menos 2 personas con imágenes de cara.")
        return

    all_pairs = positive_pairs + negative_pairs
    labeled = split_pairs(all_pairs, rng)

    n_pos = sum(1 for t in labeled if t[2] == t[3])  # person_a == person_b
    n_neg = len(labeled) - n_pos
    print(f"Total: {len(labeled)} pares ({n_pos} positivos, {n_neg} negativos).")

    write_pairs_csv(labeled, PAIRS_DIR, PROJECT_ROOT)
    print("Listo.")


if __name__ == "__main__":
    main()
