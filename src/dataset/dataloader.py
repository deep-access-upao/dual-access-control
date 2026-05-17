import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    INPUT_SHAPE,
    PAIRS_DIR,
)

TRAIN_CSV = PAIRS_DIR / "train_pairs.csv"
VAL_CSV   = PAIRS_DIR / "val_pairs.csv"
TEST_CSV  = PAIRS_DIR / "test_pairs.csv"


def load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=IMAGE_CHANNELS)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_pair(path_a: str, path_b: str, label: int):
    image_a = load_image(path_a)
    image_b = load_image(path_b)
    label_tensor = tf.cast(label, tf.float32)
    return (image_a, image_b), label_tensor


def create_pairs_dataset(
    csv_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    _require_csv(csv_path)

    df = pd.read_csv(csv_path)
    paths_a = df["image_a"].tolist()
    paths_b = df["image_b"].tolist()
    labels  = df["label"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths_a, paths_b, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda a, b, lbl: load_pair(a, b, lbl),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_train_dataset(batch_size: int = 32) -> tf.data.Dataset:
    return create_pairs_dataset(TRAIN_CSV, batch_size=batch_size, shuffle=True)


def get_val_dataset(batch_size: int = 32) -> tf.data.Dataset:
    return create_pairs_dataset(VAL_CSV, batch_size=batch_size, shuffle=False)


def get_test_dataset(batch_size: int = 32) -> tf.data.Dataset:
    return create_pairs_dataset(TEST_CSV, batch_size=batch_size, shuffle=False)


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
