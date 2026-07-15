import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.dataset.augmentations import augment_face_image
from src.config import (
    DATA_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    INPUT_SHAPE,
    PAIRS_DIR,
)

TRAIN_CSV = PAIRS_DIR / "train_pairs.csv"
VAL_CSV   = PAIRS_DIR / "val_pairs.csv"
TEST_CSV  = PAIRS_DIR / "test_pairs.csv"


def resolve_image_path(path: str) -> str:
    """Resolve manifest paths even when data lives outside the worktree."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((DATA_DIR.parent / candidate).resolve())


def load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=IMAGE_CHANNELS)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_pair(
    path_a: str,
    path_b: str,
    label: int,
    *,
    augment: bool = False,
    seed: tf.Tensor | tuple[int, int] = (42, 0),
):
    image_a = load_image(path_a)
    image_b = load_image(path_b)
    if augment:
        branch_seeds = tf.random.experimental.stateless_split(seed, num=2)
        image_a = augment_face_image(image_a, seed=branch_seeds[0])
        image_b = augment_face_image(image_b, seed=branch_seeds[1])
    label_tensor = tf.cast(label, tf.float32)
    return (image_a, image_b), label_tensor


def create_pairs_dataset(
    csv_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    seed: int = 42,
    deterministic: bool = True,
) -> tf.data.Dataset:
    """Create a pair dataset, optionally augmenting both branches independently."""
    _require_csv(csv_path)

    df = pd.read_csv(csv_path)
    paths_a = [resolve_image_path(path) for path in df["image_a"].tolist()]
    paths_b = [resolve_image_path(path) for path in df["image_b"].tolist()]
    labels  = df["label"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths_a, paths_b, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(df), 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.enumerate()
    dataset = dataset.map(
        lambda index, values: load_pair(
            values[0],
            values[1],
            values[2],
            augment=augment,
            seed=tf.stack([tf.cast(seed, tf.int32), tf.cast(index, tf.int32)]),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic,
    )
    dataset = dataset.batch(batch_size)
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_train_dataset(
    batch_size: int = 32,
    *,
    augment: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    return create_pairs_dataset(
        TRAIN_CSV,
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
        seed=seed,
    )


def get_val_dataset(batch_size: int = 32) -> tf.data.Dataset:
    return create_pairs_dataset(
        VAL_CSV, batch_size=batch_size, shuffle=False, augment=False
    )


def get_test_dataset(batch_size: int = 32) -> tf.data.Dataset:
    return create_pairs_dataset(
        TEST_CSV, batch_size=batch_size, shuffle=False, augment=False
    )


def _require_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo CSV no encontrado: {path}\n"
            "Genera los pares primero ejecutando:\n"
            "    python -m src.dataset.build_pairs"
        )


def main() -> None:
    csvs = {
        "train": TRAIN_CSV,
        "val":   VAL_CSV,
        "test":  TEST_CSV,
    }

    print("=== Par DataLoader — rutas de CSV ===")
    for split, path in csvs.items():
        estado = "OK" if path.exists() else "FALTA"
        print(f"  [{estado}] {split}: {path}")

    if not TRAIN_CSV.exists():
        print(
            "\nCSV de entrenamiento no encontrado. Genera los pares primero:\n"
            "    python -m src.dataset.build_pairs"
        )
        return

    print("\nCargando un batch de entrenamiento (batch_size=2)...")
    dataset = create_pairs_dataset(TRAIN_CSV, batch_size=2, shuffle=False)
    (batch_a, batch_b), batch_labels = next(iter(dataset))
    print(f"  forma image_a : {batch_a.shape}")
    print(f"  forma image_b : {batch_b.shape}")
    print(f"  forma labels  : {batch_labels.shape}")
    print(f"  labels        : {batch_labels.numpy()}")
    print("\nDataLoader OK.")


if __name__ == "__main__":
    main()
