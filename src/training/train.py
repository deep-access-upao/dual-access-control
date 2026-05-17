import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

from src.config import LOGS_DIR, METRICS_DIR, SAVED_MODEL_DIR
from src.dataset.dataloader import get_train_dataset, get_val_dataset
from src.models.siamese_network import build_siamese_model, compile_siamese_model

SUPPORTED_EXTENSIONS = {".keras", ".h5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de la Red Siamesa")
    parser.add_argument("--epochs",        type=int,   default=10,                    help="Número de épocas")
    parser.add_argument("--batch-size",    type=int,   default=32,                    help="Tamaño del batch")
    parser.add_argument("--learning-rate", type=float, default=0.0001,                help="Tasa de aprendizaje")
    parser.add_argument("--model-name",    type=str,   default="siamese_model.keras", help="Nombre del archivo del modelo guardado")
    args = parser.parse_args()
    args.model_name = _normalize_model_name(args.model_name)
    return args


def _normalize_model_name(name: str) -> str:
    # Si la extensión no es soportada, se fuerza .keras como formato por defecto
    if Path(name).suffix not in SUPPORTED_EXTENSIONS:
        name = Path(name).stem + ".keras"
    return name


def ensure_output_directories() -> None:
    for directory in (SAVED_MODEL_DIR, LOGS_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_callbacks(model_path: Path, log_path: Path) -> list:
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(log_path),
        append=False,
    )
    return [checkpoint, early_stopping, csv_logger]


def save_history(history: dict, metrics_dir: Path, model_name: str) -> None:
    # Convierte tensores numpy a tipos Python nativos para que json.dump funcione
    serializable = {key: [float(v) for v in values] for key, values in history.items()}
    stem = model_name.replace(".keras", "").replace(".h5", "")
    output_path = metrics_dir / f"{stem}_history.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Historial guardado en: {output_path}")


def train_model(args: argparse.Namespace) -> None:
    ensure_output_directories()

    model_path = SAVED_MODEL_DIR / args.model_name
    log_path   = LOGS_DIR / (args.model_name.replace(".keras", "").replace(".h5", "") + "_training.csv")

    # Los datasets se cargan primero para detectar CSVs faltantes antes de construir el modelo
    print("Cargando datasets...")
    train_dataset = get_train_dataset(batch_size=args.batch_size)
    val_dataset   = get_val_dataset(batch_size=args.batch_size)

    print("\nConstruyendo el modelo siamés...")
    model = build_siamese_model()
    model = compile_siamese_model(model, learning_rate=args.learning_rate)
    model.summary()

    callbacks = create_callbacks(model_path, log_path)

    print(f"\nIniciando entrenamiento — épocas: {args.epochs}, batch: {args.batch_size}, lr: {args.learning_rate}\n")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Guarda el modelo final (puede diferir del mejor checkpoint si no hubo mejora al final)
    final_path = SAVED_MODEL_DIR / ("final_" + args.model_name)
    model.save(str(final_path))
    print(f"\nModelo final guardado en: {final_path}")

    save_history(history.history, METRICS_DIR, args.model_name)


def main() -> None:
    args = parse_args()
    try:
        train_model(args)
    except FileNotFoundError as error:
        print(f"\nError: {error}")
        print("\nGenera los pares primero ejecutando:")
        print("    python -m src.dataset.build_pairs")
        sys.exit(1)


if __name__ == "__main__":
    main()
