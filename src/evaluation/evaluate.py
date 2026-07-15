"""Calibra el umbral con validation y evalúa una sola vez el test limpio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import EXPERIMENTS_DIR, SAVED_MODEL_DIR
from src.dataset.dataloader import (
    TEST_CSV,
    VAL_CSV,
    get_test_dataset,
    get_val_dataset,
)
from src.models.siamese_network import l1_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibración y evaluación formal del baseline")
    parser.add_argument("--mode", choices=("calibrate", "test"), required=True)
    parser.add_argument(
        "--experiment-name", default="baseline_formal/baseline_con_aumento"
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--criterion",
        choices=("max_f1",),
        default="max_f1",
        help="Criterio aplicado exclusivamente a validation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Umbral explícito para test; por defecto se lee threshold.json",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    model_path = args.model_path or SAVED_MODEL_DIR / f"{args.experiment_name}.keras"
    return Path(model_path), experiment_dir


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        str(model_path), custom_objects={"l1_distance": l1_distance}, compile=False
    )


def collect_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    scores_batches, labels_batches = [], []
    for (image_a, image_b), labels in dataset:
        scores_batches.append(np.asarray(model.predict_on_batch([image_a, image_b])).reshape(-1))
        labels_batches.append(np.asarray(labels).reshape(-1))
    if not scores_batches:
        return np.array([]), np.array([])
    return (
        np.concatenate(labels_batches).astype(np.int32),
        np.concatenate(scores_batches).astype(np.float32),
    )


def compute_far_frr(cm: np.ndarray) -> Tuple[float, float]:
    tn, fp, fn, tp = cm.ravel()
    far = float(fp / (fp + tn)) if fp + tn else 0.0
    frr = float(fn / (fn + tp)) if fn + tp else 0.0
    return far, frr


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    far, frr = compute_far_frr(cm)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "num_pairs": int(len(y_true)),
        "num_positives": int(np.sum(y_true == 1)),
        "num_negatives": int(np.sum(y_true == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": far,
        "frr": frr,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    candidates = np.unique(np.concatenate(([0.0], y_score.astype(float), [1.0])))
    rows = []
    for threshold in candidates:
        metrics = compute_metrics(y_true, y_score, float(threshold))
        rows.append({key: metrics[key] for key in ("threshold", "accuracy", "precision", "recall", "f1", "far", "frr")})
    return pd.DataFrame(rows)


def select_threshold(search: pd.DataFrame, criterion: str = "max_f1") -> pd.Series:
    if criterion != "max_f1":
        raise ValueError(f"Criterio no soportado: {criterion}")
    # En empates de F1 se prioriza menor FAR y luego menor FRR, coherente con control de acceso.
    return search.sort_values(
        ["f1", "far", "frr", "threshold"],
        ascending=[False, True, True, False],
        kind="stable",
    ).iloc[0]


def save_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_predictions(y_true, y_score, threshold: float, path: Path) -> None:
    pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "y_score": y_score.astype(float),
            "y_pred": (y_score >= threshold).astype(int),
        }
    ).to_csv(path, index=False)


def plot_confusion(metrics: dict, title: str, path: Path) -> None:
    cm_dict = metrics["confusion_matrix"]
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(title=title, xlabel="Predicción", ylabel="Etiqueta real")
    ax.set_xticks([0, 1], ["Diferente", "Misma persona"])
    ax.set_yticks([0, 1], ["Diferente", "Misma persona"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(cm[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, prefix: str, output_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title=f"ROC — {prefix}", xlabel="FAR", ylabel="TPR (1 - FRR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=140)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set(title=f"Precision-Recall — {prefix}", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_precision_recall_curve.png", dpi=140)
    plt.close(fig)


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    for key in ("num_pairs", "num_positives", "num_negatives", "threshold", "accuracy", "precision", "recall", "f1", "far", "frr", "roc_auc"):
        print(f"{key:16s}: {metrics[key]}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")


def calibrate(model: tf.keras.Model, experiment_dir: Path, batch_size: int, criterion: str) -> dict:
    print(f"CSV de calibration (validation, sin aumentos): {VAL_CSV}")
    y_true, y_score = collect_predictions(model, get_val_dataset(batch_size=batch_size))
    if y_true.size == 0:
        raise ValueError("Validation está vacío")
    search = threshold_sweep(y_true, y_score)
    selected = select_threshold(search, criterion)
    threshold = float(selected["threshold"])
    metrics = compute_metrics(y_true, y_score, threshold)
    metrics.update(
        {
            "split": "validation",
            "threshold_selection_criterion": "maximizar F1; desempatar por menor FAR y luego menor FRR",
            "threshold_selected_with_test": False,
        }
    )
    search.to_csv(experiment_dir / "threshold_search.csv", index=False)
    save_json(
        {
            "threshold": threshold,
            "criterion": metrics["threshold_selection_criterion"],
            "calibration_split": "validation",
            "test_used": False,
        },
        experiment_dir / "threshold.json",
    )
    save_json(metrics, experiment_dir / "validation_metrics.json")
    save_predictions(y_true, y_score, threshold, experiment_dir / "validation_predictions.csv")
    plot_confusion(metrics, "Matriz de confusión — validation", experiment_dir / "validation_confusion_matrix.png")
    plot_roc_pr(y_true, y_score, "validation", experiment_dir)
    print_metrics("Calibración en validation", metrics)
    return metrics


def load_calibrated_threshold(experiment_dir: Path) -> float:
    path = experiment_dir / "threshold.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path}. Calibra primero con --mode calibrate; no se permite elegir el umbral con test."
        )
    with path.open(encoding="utf-8") as file:
        return float(json.load(file)["threshold"])


def evaluate_test(model: tf.keras.Model, experiment_dir: Path, batch_size: int, threshold: float) -> dict:
    print(f"CSV de test limpio (sin aumentos): {TEST_CSV}")
    y_true, y_score = collect_predictions(model, get_test_dataset(batch_size=batch_size))
    if y_true.size == 0:
        raise ValueError("Test está vacío")
    metrics = compute_metrics(y_true, y_score, threshold)
    metrics.update(
        {
            "split": "test_clean",
            "threshold_source": "validation",
            "random_augmentation": False,
        }
    )
    save_json(metrics, experiment_dir / "test_metrics.json")
    save_predictions(y_true, y_score, threshold, experiment_dir / "test_predictions.csv")
    plot_confusion(metrics, "Matriz de confusión — test limpio", experiment_dir / "test_confusion_matrix.png")
    plot_roc_pr(y_true, y_score, "test_clean", experiment_dir)
    print_metrics("Evaluación final en test limpio", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    model_path, experiment_dir = resolve_paths(args)
    required_csv = VAL_CSV if args.mode == "calibrate" else TEST_CSV
    if not model_path.is_file() or not required_csv.is_file():
        print(f"ERROR: falta modelo o CSV: {model_path}; {required_csv}")
        sys.exit(1)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path)
    try:
        if args.mode == "calibrate":
            calibrate(model, experiment_dir, args.batch_size, args.criterion)
        else:
            threshold = args.threshold if args.threshold is not None else load_calibrated_threshold(experiment_dir)
            evaluate_test(model, experiment_dir, args.batch_size, threshold)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
