"""Evaluación determinista de robustez, separada del test limpio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import EXPERIMENTS_DIR, SAVED_MODEL_DIR
from src.dataset.dataloader import TEST_CSV, load_image, resolve_image_path
from src.evaluation.evaluate import compute_metrics, load_calibrated_threshold, load_model
from src.evaluation.stress_tests import STRESS_CONDITIONS, apply_stress_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pruebas de estrés deterministas del baseline")
    parser.add_argument(
        "--experiment-name", default="baseline_formal/baseline_con_aumento"
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def create_stress_dataset(
    csv_path: Path,
    condition: str,
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    frame = pd.read_csv(csv_path)
    paths_a = [resolve_image_path(path) for path in frame["image_a"]]
    paths_b = [resolve_image_path(path) for path in frame["image_b"]]
    labels = frame["label"].astype(int).tolist()
    condition_index = STRESS_CONDITIONS.index(condition)
    dataset = tf.data.Dataset.from_tensor_slices((paths_a, paths_b, labels)).enumerate()

    def load_stressed_pair(index, values):
        image_a = load_image(values[0])
        image_b = load_image(values[1])
        stress_seed = tf.stack(
            [tf.cast(seed + condition_index, tf.int32), tf.cast(index, tf.int32)]
        )
        image_b = apply_stress_condition(image_b, condition, seed=stress_seed)
        return (image_a, image_b), tf.cast(values[2], tf.float32)

    dataset = dataset.map(
        load_stressed_pair,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.batch(batch_size).with_options(options).prefetch(tf.data.AUTOTUNE)


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    labels, scores = [], []
    for (image_a, image_b), batch_labels in dataset:
        labels.append(np.asarray(batch_labels).reshape(-1))
        scores.append(np.asarray(model.predict_on_batch([image_a, image_b])).reshape(-1))
    return np.concatenate(labels).astype(np.int32), np.concatenate(scores).astype(np.float32)


def plot_stress_summary(frame: pd.DataFrame, path: Path) -> None:
    positions = np.arange(len(frame))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(positions - width, frame["accuracy"], width, label="Accuracy")
    ax.bar(positions, frame["far"], width, label="FAR")
    ax.bar(positions + width, frame["frr"], width, label="FRR")
    ax.set_xticks(positions, frame["condition"], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Robustez por condición (probe/image_b alterada)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_stress(
    model: tf.keras.Model,
    experiment_dir: Path,
    threshold: float,
    batch_size: int,
    seed: int,
) -> list[dict]:
    results = []
    for condition in STRESS_CONDITIONS:
        print(f"Evaluando estrés: {condition}")
        dataset = create_stress_dataset(TEST_CSV, condition, batch_size, seed)
        y_true, y_score = collect_predictions(model, dataset)
        metrics = compute_metrics(y_true, y_score, threshold)
        metrics.update(
            {
                "condition": condition,
                "split": "test_stress_separate",
                "stressed_branch": "image_b (probe)",
                "threshold_source": "validation",
                "deterministic_seed": seed,
            }
        )
        results.append(metrics)
        print(
            f"  accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"FAR={metrics['far']:.4f} FRR={metrics['frr']:.4f}"
        )

    with (experiment_dir / "stress_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    columns = [
        "condition",
        "num_pairs",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "far",
        "frr",
        "roc_auc",
        "threshold",
    ]
    frame = pd.DataFrame(results)
    frame[columns].to_csv(experiment_dir / "stress_metrics.csv", index=False)
    plot_stress_summary(frame, experiment_dir / "stress_summary.png")
    return results


def main() -> None:
    args = parse_args()
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    model_path = args.model_path or SAVED_MODEL_DIR / f"{args.experiment_name}.keras"
    if not Path(model_path).is_file() or not TEST_CSV.is_file():
        print(f"ERROR: falta modelo o test CSV: {model_path}; {TEST_CSV}")
        sys.exit(1)
    try:
        threshold = load_calibrated_threshold(experiment_dir)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    model = load_model(Path(model_path))
    evaluate_stress(model, experiment_dir, threshold, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
