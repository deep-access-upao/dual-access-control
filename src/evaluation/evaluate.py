import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    METRICS_DIR,
    PLOTS_DIR,
    SAVED_MODEL_DIR,
)
from src.dataset.dataloader import TEST_CSV, get_test_dataset


METRICS_REPORT_FILENAME = "evaluation_report.json"
PREDICTIONS_FILENAME = "test_predictions.csv"
CONFUSION_MATRIX_PLOT = "confusion_matrix.png"
ROC_CURVE_PLOT = "roc_curve.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de la Red Siamesa")
    parser.add_argument("--model-path", type=str, default=None,                          help="Ruta directa al modelo entrenado")
    parser.add_argument("--model-name", type=str, default="siamese_model.keras",         help="Nombre del archivo del modelo dentro de SAVED_MODEL_DIR")
    parser.add_argument("--batch-size", type=int, default=32,                            help="Tamaño del batch para la evaluación")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Umbral de similitud para clasificar como GRANTED")
    return parser.parse_args()


def resolve_model_path(model_path: str | None, model_name: str) -> Path:
    if model_path:
        return Path(model_path)
    return SAVED_MODEL_DIR / model_name


def ensure_output_directories() -> None:
    for directory in (METRICS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    # Se itera manualmente para conservar y_true alineado con y_score batch a batch
    scores_batches = []
    labels_batches = []

    for (image_a, image_b), labels in dataset:
        batch_scores = model.predict_on_batch([image_a, image_b])
        scores_batches.append(np.asarray(batch_scores).reshape(-1))
        labels_batches.append(np.asarray(labels).reshape(-1))

    if not scores_batches:
        return np.array([]), np.array([])

    y_score = np.concatenate(scores_batches).astype(np.float32)
    y_true = np.concatenate(labels_batches).astype(np.int32)
    return y_true, y_score


def compute_far_frr(cm: np.ndarray) -> Tuple[float, float]:
    # cm tiene forma [[TN, FP], [FN, TP]] según el orden de clases [0, 1]
    tn, fp, fn, tp = cm.ravel()

    # FAR: proporción de impostores aceptados sobre el total de impostores reales
    far_denominator = fp + tn
    far = float(fp) / float(far_denominator) if far_denominator > 0 else 0.0

    # FRR: proporción de genuinos rechazados sobre el total de genuinos reales
    frr_denominator = fn + tp
    frr = float(fn) / float(frr_denominator) if frr_denominator > 0 else 0.0

    return far, frr


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    y_pred = (y_score >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    far, frr = compute_far_frr(cm)

    # ROC AUC solo es válido cuando hay ambas clases presentes en y_true
    both_classes_present = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, y_score)) if both_classes_present else None

    metrics_dict = {
        "threshold": float(threshold),
        "num_samples": int(len(y_true)),
        "num_positives": int(np.sum(y_true == 1)),
        "num_negatives": int(np.sum(y_true == 0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "far": far,
        "frr": frr,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negative":  int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive":  int(cm[1, 1]),
        },
    }

    return metrics_dict, y_pred, cm


def save_metrics(metrics_dict: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Métricas guardadas en: {output_path}")


def save_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    df = pd.DataFrame({
        "y_true":  y_true.astype(int),
        "y_score": y_score.astype(float),
        "y_pred":  y_pred.astype(int),
    })
    df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["DENIED (0)", "GRANTED (1)"])
    ax.set_yticklabels(["DENIED (0)", "GRANTED (1)"])

    # Etiquetas numéricas dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
    ax.set_title("Curva ROC")
    ax.set_xlabel("Tasa de falsos positivos (FAR)")
    ax.set_ylabel("Tasa de verdaderos positivos (1 - FRR)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Curva ROC guardada en: {output_path}")


def _print_summary(metrics_dict: dict) -> None:
    print("\n=== Resultados de la evaluación ===")
    print(f"  muestras       : {metrics_dict['num_samples']} "
          f"(positivas: {metrics_dict['num_positives']}, "
          f"negativas: {metrics_dict['num_negatives']})")
    print(f"  umbral         : {metrics_dict['threshold']:.4f}")
    print(f"  accuracy       : {metrics_dict['accuracy']:.4f}")
    print(f"  precision      : {metrics_dict['precision']:.4f}")
    print(f"  recall         : {metrics_dict['recall']:.4f}")
    print(f"  f1_score       : {metrics_dict['f1_score']:.4f}")
    print(f"  FAR            : {metrics_dict['far']:.4f}")
    print(f"  FRR            : {metrics_dict['frr']:.4f}")
    roc_auc = metrics_dict["roc_auc"]
    print(f"  ROC AUC        : {roc_auc:.4f}" if roc_auc is not None else "  ROC AUC        : N/A (solo una clase en y_true)")


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_path, args.model_name)
    if not model_path.exists():
        print(f"Modelo no encontrado en: {model_path}")
        print("Entrena el modelo primero ejecutando:")
        print("    python -m src.training.train")
        sys.exit(1)

    if not TEST_CSV.exists():
        print(f"CSV de test no encontrado en: {TEST_CSV}")
        print("Genera los pares primero ejecutando:")
        print("    python -m src.dataset.build_pairs")
        sys.exit(1)

    ensure_output_directories()

    print(f"Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print("Cargando dataset de test...")
    test_dataset = get_test_dataset(batch_size=args.batch_size)

    print("Generando predicciones...")
    y_true, y_score = collect_predictions(model, test_dataset)

    if y_true.size == 0:
        print("El dataset de test está vacío. No hay nada que evaluar.")
        sys.exit(1)

    metrics_dict, y_pred, cm = compute_metrics(y_true, y_score, args.threshold)
    _print_summary(metrics_dict)

    save_metrics(metrics_dict, METRICS_DIR / METRICS_REPORT_FILENAME)
    save_predictions(y_true, y_score, y_pred, METRICS_DIR / PREDICTIONS_FILENAME)
    plot_confusion_matrix(cm, PLOTS_DIR / CONFUSION_MATRIX_PLOT)

    # La curva ROC requiere ambas clases para ser interpretable
    if len(np.unique(y_true)) == 2:
        plot_roc_curve(y_true, y_score, PLOTS_DIR / ROC_CURVE_PLOT)
    else:
        print("Curva ROC omitida: y_true solo contiene una clase.")


if __name__ == "__main__":
    main()
