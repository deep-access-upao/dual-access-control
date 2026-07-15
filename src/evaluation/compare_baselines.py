"""Genera una comparación reproducible de los dos baselines formales."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EXPERIMENTS_DIR


FORMAL_DIR = EXPERIMENTS_DIR / "baseline_formal"
OUTPUT_DIR = FORMAL_DIR / "comparison"
VARIANTS = {
    "con_aumento": FORMAL_DIR / "baseline_con_aumento",
    "sin_aumento": FORMAL_DIR / "baseline_sin_aumento",
}
METRICS = ("accuracy", "precision", "recall", "f1", "far", "frr", "roc_auc")


def read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_results() -> dict[str, dict]:
    results = {}
    for name, directory in VARIANTS.items():
        results[name] = {
            "validation": read_json(directory / "validation_metrics.json"),
            "test_clean": read_json(directory / "test_metrics.json"),
            "stress": read_json(directory / "stress_metrics.json"),
        }
    return results


def build_summary(results: dict[str, dict]) -> dict:
    summary = {"variants": results}
    for name, result in results.items():
        stress = result["stress"]
        worst = min(stress, key=lambda row: (row["accuracy"], row["f1"]))
        summary.setdefault("stress_summary", {})[name] = {
            "mean_accuracy": float(np.mean([row["accuracy"] for row in stress])),
            "mean_f1": float(np.mean([row["f1"] for row in stress])),
            "mean_far": float(np.mean([row["far"] for row in stress])),
            "mean_frr": float(np.mean([row["frr"] for row in stress])),
            "worst_condition": worst["condition"],
            "worst_accuracy": worst["accuracy"],
            "worst_f1": worst["f1"],
        }
    summary["recommended_variant"] = "con_aumento"
    summary["recommendation_basis"] = (
        "Mejor accuracy/F1 promedio y peor caso en stress, con rendimiento limpio similar. "
        "Debe vigilarse el FAR limpio y reforzarse poca luz con datos reales."
    )
    return summary


def write_clean_test_csv(results: dict[str, dict]) -> None:
    path = OUTPUT_DIR / "clean_test_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("variant", "threshold", *METRICS, "tn", "fp", "fn", "tp"),
        )
        writer.writeheader()
        for name, result in results.items():
            test = result["test_clean"]
            writer.writerow(
                {
                    "variant": name,
                    "threshold": test["threshold"],
                    **{metric: test[metric] for metric in METRICS},
                    **test["confusion_matrix"],
                }
            )


def plot_clean_test(results: dict[str, dict]) -> None:
    names = list(results)
    labels = ["Con aumento", "Sin aumento"]
    metrics = ("accuracy", "f1", "far", "frr")
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, name in enumerate(names):
        values = [results[name]["test_clean"][metric] for metric in metrics]
        ax.bar(x + (index - 0.5) * width, values, width, label=labels[index])
    ax.set_xticks(x, ["Accuracy", "F1", "FAR", "FRR"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparación en test limpio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clean_test_comparison.png", dpi=140)
    plt.close(fig)


def plot_stress(results: dict[str, dict]) -> None:
    conditions = [row["condition"] for row in results["con_aumento"]["stress"]]
    labels = [condition.replace("_", "\n") for condition in conditions]
    x = np.arange(len(conditions))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, metric, title in zip(axes, ("accuracy", "f1"), ("Accuracy", "F1")):
        for index, (name, display_name) in enumerate(
            (("con_aumento", "Con aumento"), ("sin_aumento", "Sin aumento"))
        ):
            by_condition = {row["condition"]: row[metric] for row in results[name]["stress"]}
            axis.bar(
                x + (index - 0.5) * width,
                [by_condition[condition] for condition in conditions],
                width,
                label=display_name,
            )
        axis.set_ylabel(title)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    axes[-1].set_xticks(x, labels, fontsize=8)
    fig.suptitle("Robustez por condición de stress")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stress_comparison.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    summary = build_summary(results)
    with (OUTPUT_DIR / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    write_clean_test_csv(results)
    plot_clean_test(results)
    plot_stress(results)
    print(f"Comparación guardada en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
