"""Entrenamiento reproducible del baseline siamés con validación previa obligatoria."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import (
    DATASET_MANIFEST_PATH,
    EXPERIMENTS_DIR,
    PAIRS_DIR,
    SAVED_MODEL_DIR,
)
from src.dataset.audit_splits import audit, print_report
from src.dataset.dataloader import TRAIN_CSV, VAL_CSV, get_train_dataset, get_val_dataset
from src.dataset.validate_support_set import validate_support_set
from src.models.siamese_network import (
    build_siamese_model,
    compile_siamese_model,
    l1_distance,
)
from src.utils.tensorflow_runtime import (
    SUPPORTED_DEVICES,
    configure_tensorflow_runtime,
)

SUPPORTED_EXTENSIONS = {".keras", ".h5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento formal de la red siamesa baseline")
    parser.add_argument("--epochs", type=int, default=10, help="Máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Tasa de aprendizaje")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia de early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de Python, NumPy y TensorFlow")
    parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICES,
        default="auto",
        help="Dispositivo TensorFlow: auto detecta GPU, cpu la desactiva y gpu la exige",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline_formal/baseline_con_aumento",
        help="Subcarpeta dentro de outputs/experiments y models/saved_model",
    )
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        help="Activa aumentos realistas solo en train (valor por defecto)",
    )
    augmentation.add_argument(
        "--no-augmentation",
        dest="augmentation",
        action="store_false",
        help="Desactiva aumentos en train",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda desde el checkpoint y el history.csv del experimento",
    )
    parser.set_defaults(augmentation=True)
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run_preflight_checks() -> None:
    """Block training when split leakage or support-set errors are detected."""
    print("\n=== Validación obligatoria previa al entrenamiento ===")
    report, errors = audit(DATASET_MANIFEST_PATH, PAIRS_DIR)
    print_report(report, errors)
    if errors:
        raise RuntimeError("La auditoría detectó fuga; el entrenamiento fue cancelado.")
    if validate_support_set(strict=False, manifest_path=DATASET_MANIFEST_PATH) != 0:
        raise RuntimeError("La validación del support set falló; el entrenamiento fue cancelado.")


def create_callbacks(
    model_path: Path,
    history_csv: Path,
    patience: int,
    *,
    append_history: bool = False,
    initial_best_val_loss: float | None = None,
) -> list:
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path), monitor="val_loss", save_best_only=True, verbose=1
    )
    if initial_best_val_loss is not None:
        checkpoint.best = initial_best_val_loss
    return [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            baseline=initial_best_val_loss,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(filename=str(history_csv), append=append_history),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def read_history(history_csv: Path) -> list[dict[str, str]]:
    if not history_csv.is_file():
        return []
    with history_csv.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_history(history_csv: Path, experiment_dir: Path) -> list[dict[str, str]]:
    rows = read_history(history_csv)
    metric_names = [key for key in rows[0] if key != "epoch"] if rows else []
    serializable = {
        key: [float(row[key]) for row in rows]
        for key in metric_names
    }
    with (experiment_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)

    epochs = [int(row["epoch"]) + 1 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, serializable.get("loss", []), label="train")
    axes[0].plot(epochs, serializable.get("val_loss", []), label="validation")
    axes[0].set(title="Loss por época", xlabel="Época", ylabel="Binary cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, serializable.get("binary_accuracy", []), label="train")
    axes[1].plot(epochs, serializable.get("val_binary_accuracy", []), label="validation")
    axes[1].set(title="Accuracy por época", xlabel="Época", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(experiment_dir / "training_history.png", dpi=140)
    plt.close(fig)
    return rows


def resume_state(model_path: Path, history_csv: Path) -> tuple[tf.keras.Model, int, float]:
    rows = read_history(history_csv)
    if not model_path.is_file() or not rows:
        raise FileNotFoundError(
            "No se puede reanudar: falta el checkpoint o el history.csv del experimento."
        )
    initial_epoch = int(rows[-1]["epoch"]) + 1
    best_val_loss = min(float(row["val_loss"]) for row in rows)
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"l1_distance": l1_distance},
        compile=True,
    )
    return model, initial_epoch, best_val_loss


def training_config(
    args: argparse.Namespace,
    model_path: Path,
    initial_epoch: int,
) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": args.experiment_name,
        "architecture": "baseline siamés actual (sin cambios estructurales)",
        "train_csv": "data/pairs/train_pairs.csv",
        "validation_csv": "data/pairs/val_pairs.csv",
        "train_augmentation": bool(args.augmentation),
        "validation_augmentation": False,
        "test_used_during_training": False,
        "resume": bool(args.resume),
        "initial_epoch": initial_epoch,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping": {"monitor": "val_loss", "patience": args.patience},
        "checkpoint": {"monitor": "val_loss", "save_best_only": True},
        "seed": args.seed,
        "tensorflow_version": tf.__version__,
        "device": getattr(args, "device", "auto"),
        "model_path": f"models/saved_model/{args.experiment_name}.keras",
    }


def train_model(args: argparse.Namespace) -> Path:
    configure_tensorflow_runtime(getattr(args, "device", "auto"))
    set_reproducible_seed(args.seed)
    run_preflight_checks()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    model_path = SAVED_MODEL_DIR / f"{args.experiment_name}.keras"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv = experiment_dir / "history.csv"

    if args.resume:
        model, initial_epoch, best_val_loss = resume_state(model_path, history_csv)
        if initial_epoch >= args.epochs:
            raise ValueError(
                f"El experimento ya tiene {initial_epoch} épocas; --epochs={args.epochs} no agrega trabajo."
            )
    else:
        model = compile_siamese_model(
            build_siamese_model(), learning_rate=args.learning_rate
        )
        initial_epoch = 0
        best_val_loss = None

    config = training_config(args, model_path, initial_epoch)
    with (experiment_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print("\n=== Configuración de entrenamiento ===")
    print(f"Train CSV              : {TRAIN_CSV}")
    print(f"Validation CSV         : {VAL_CSV}")
    print(f"Train augmentation     : {args.augmentation}")
    print("Validation augmentation: False")
    print(f"Épocas/batch/lr        : {args.epochs}/{args.batch_size}/{args.learning_rate}")
    print(f"Early stopping         : val_loss, patience={args.patience}")
    print(f"Semilla                : {args.seed}")
    print(f"Reanudación            : {args.resume} (época inicial={initial_epoch})")
    print(f"Modelo                  : {model_path}")
    print(f"Resultados              : {experiment_dir}")

    train_dataset = get_train_dataset(
        batch_size=args.batch_size, augment=args.augmentation, seed=args.seed
    )
    validation_dataset = get_val_dataset(batch_size=args.batch_size)

    model.summary()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=create_callbacks(
            model_path,
            history_csv,
            args.patience,
            append_history=args.resume,
            initial_best_val_loss=best_val_loss,
        ),
    )
    rows = save_history(history_csv, experiment_dir)
    completed = len(rows)
    validation_losses = [float(row["val_loss"]) for row in rows]
    config["epochs_completed"] = completed
    config["stopped_reason"] = (
        "early_stopping" if completed < args.epochs else "epochs_max_reached"
    )
    if validation_losses:
        best_index = int(np.argmin(validation_losses))
        config["checkpoint"].update(
            {
                "best_epoch": best_index + 1,
                "best_val_loss": float(validation_losses[best_index]),
            }
        )
    with (experiment_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    print(f"\nEntrenamiento finalizado. Mejor modelo guardado en: {model_path}")
    return model_path


def main() -> None:
    args = parse_args()
    try:
        train_model(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
