"""Compara baseline y GAP+L2+coseno usando resultados ya fijados."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_DIR

BASELINE_NAME = "baseline_con_aumento"
VARIANT_NAME = "siamese_gap_l2_cosine"
MODEL_DIRS = {
    BASELINE_NAME: EXPERIMENTS_DIR / "baseline_formal" / BASELINE_NAME,
    VARIANT_NAME: EXPERIMENTS_DIR / "baseline_formal" / VARIANT_NAME,
}
OUTPUT_DIR = EXPERIMENTS_DIR / "model_comparison"
PHOTOMETRIC_CONDITIONS = ("low_light", "overexposure", "low_contrast")
METRICS = ("accuracy", "precision", "recall", "f1", "far", "frr")


def _calibration_dir(model_dir: Path) -> Path:
    return model_dir / "security_threshold_calibration"


def load_candidate_table(model_dir: Path) -> pd.DataFrame:
    path = _calibration_dir(model_dir) / "security_threshold_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Falta la calibración requerida: {path}")
    frame = pd.read_csv(path)
    required = {"criterion", "threshold"}
    for split in ("validation", "test"):
        required.update(f"{split}_{metric}" for metric in METRICS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}")
    return frame


def load_stress_table(model_dir: Path) -> pd.DataFrame:
    path = _calibration_dir(model_dir) / "stress_metrics_by_candidate.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Faltan resultados de stress: {path}")
    frame = pd.read_csv(path)
    required = {"criterion", "condition", "accuracy", "f1", "far", "frr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}")
    return frame


def candidate_record(model: str, row: pd.Series) -> dict:
    record = {
        "model": model,
        "criterion": str(row["criterion"]),
        "threshold": float(row["threshold"]),
    }
    for split in ("validation", "test"):
        for metric in METRICS:
            record[f"{split}_{metric}"] = float(row[f"{split}_{metric}"])
        for cell in ("tn", "fp", "fn", "tp"):
            record[f"{split}_{cell}"] = int(row[f"{split}_{cell}"])
    record["test_roc_auc"] = float(row["test_roc_auc"])
    return record


def stress_records(model: str, frame: pd.DataFrame) -> list[dict]:
    selected = frame[frame["criterion"] == "security_first"]
    if selected.empty:
        raise ValueError(f"{model}: falta el candidato security_first en stress")
    return [
        {
            "model": model,
            "criterion": "security_first",
            "threshold": float(row.threshold),
            "condition": str(row.condition),
            "accuracy": float(row.accuracy),
            "f1": float(row.f1),
            "far": float(row.far),
            "frr": float(row.frr),
        }
        for row in selected.itertuples(index=False)
    ]


def recommend_model(
    baseline: dict,
    variant: dict,
    baseline_stress: dict[str, dict],
    variant_stress: dict[str, dict],
    *,
    validation_stress_available: bool = False,
) -> dict:
    """Aplica de forma explícita los criterios de aceptación de la sesión."""
    clean_far_not_worse = (
        variant["validation_far"] <= baseline["validation_far"] + 1e-12
    )
    clean_quality_ok = (
        variant["validation_f1"] >= baseline["validation_f1"] - 0.01
        and variant["validation_accuracy"]
        >= baseline["validation_accuracy"] - 0.01
    )
    improvements = {
        condition: baseline_stress[condition]["frr"]
        - variant_stress[condition]["frr"]
        for condition in PHOTOMETRIC_CONDITIONS
    }
    materially_improved = [
        condition for condition, delta in improvements.items() if delta >= 0.05 - 1e-12
    ]
    far_controlled = all(
        variant_stress[condition]["far"]
        <= baseline_stress[condition]["far"] + 0.02 + 1e-12
        for condition in materially_improved
    )
    target_improvement_validated = bool(
        validation_stress_available and materially_improved and far_controlled
    )
    replace = bool(
        clean_far_not_worse
        and clean_quality_ok
        and target_improvement_validated
    )
    return {
        "selected_model": VARIANT_NAME if replace else BASELINE_NAME,
        "replace_baseline": replace,
        "criteria": {
            "validation_clean_far_not_worse": clean_far_not_worse,
            "validation_clean_f1_accuracy_within_one_point": clean_quality_ok,
            "validation_stress_available": validation_stress_available,
            "target_improvement_validated": target_improvement_validated,
        },
        "secondary_test_stress": {
            "photometric_frr_improvement_at_least_five_points": materially_improved,
            "photometric_far_controlled": far_controlled,
            "frr_delta_baseline_minus_variant": improvements,
        },
        "stress_role": (
            "Test-stress es evaluación secundaria: no calibra el threshold, no "
            "demuestra mejora en validation y no habilita reemplazar el baseline."
        ),
    }


def plot_clean(security: pd.DataFrame) -> None:
    positions = np.arange(len(security))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, column, label in (
        (-1.5, "test_accuracy", "Accuracy"),
        (-0.5, "test_f1", "F1"),
        (0.5, "test_far", "FAR"),
        (1.5, "test_frr", "FRR"),
    ):
        ax.bar(positions + offset * width, security[column], width, label=label)
    ax.set_xticks(positions, security["model"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Test limpio con security_first")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clean_test_comparison.png", dpi=140)
    plt.close(fig)


def plot_far_frr(security: pd.DataFrame) -> None:
    positions = np.arange(len(security))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(positions - width / 2, security["test_far"], width, label="FAR")
    ax.bar(positions + width / 2, security["test_frr"], width, label="FRR")
    ax.set_xticks(positions, security["model"])
    ax.set_ylim(0, max(0.1, float(security[["test_far", "test_frr"]].max().max()) * 1.25))
    ax.set_title("FAR/FRR limpio con security_first")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clean_far_frr_comparison.png", dpi=140)
    plt.close(fig)


def plot_photometric(stress: pd.DataFrame) -> None:
    selected = stress[stress["condition"].isin(PHOTOMETRIC_CONDITIONS)]
    positions = np.arange(len(PHOTOMETRIC_CONDITIONS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, model in enumerate((BASELINE_NAME, VARIANT_NAME)):
        by_condition = selected[selected["model"] == model].set_index("condition")
        values = [float(by_condition.loc[condition, "frr"]) for condition in PHOTOMETRIC_CONDITIONS]
        ax.bar(positions + (index - 0.5) * width, values, width, label=model)
    ax.set_xticks(positions, PHOTOMETRIC_CONDITIONS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("FRR")
    ax.set_title("FRR bajo degradación fotométrica")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "photometric_frr_comparison.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_rows = []
    stress_rows = []
    for model, directory in MODEL_DIRS.items():
        candidates = load_candidate_table(directory)
        for criterion in ("max_f1", "security_first"):
            selected = candidates[candidates["criterion"] == criterion]
            if selected.empty:
                raise ValueError(f"{model}: falta el criterio {criterion}")
            threshold_rows.append(candidate_record(model, selected.iloc[0]))
        stress_rows.extend(stress_records(model, load_stress_table(directory)))

    thresholds = pd.DataFrame(threshold_rows)
    stress = pd.DataFrame(stress_rows)
    security = thresholds[thresholds["criterion"] == "security_first"].copy()
    stress_maps = {
        model: stress[stress["model"] == model].set_index("condition").to_dict("index")
        for model in MODEL_DIRS
    }
    records = security.set_index("model").to_dict("index")
    decision = recommend_model(
        records[BASELINE_NAME],
        records[VARIANT_NAME],
        stress_maps[BASELINE_NAME],
        stress_maps[VARIANT_NAME],
        validation_stress_available=False,
    )

    thresholds.to_csv(OUTPUT_DIR / "threshold_metrics.csv", index=False)
    stress.to_csv(OUTPUT_DIR / "stress_metrics_security_first.csv", index=False)
    photometric = stress[stress["condition"].isin(PHOTOMETRIC_CONDITIONS)]
    photometric.to_csv(OUTPUT_DIR / "photometric_metrics.csv", index=False)
    with (OUTPUT_DIR / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_basis": "validation; test y test-stress solo como evaluación final",
                "validation_stress_available": False,
                "threshold_metrics": threshold_rows,
                "stress_metrics_security_first": stress_rows,
                "decision": decision,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    plot_clean(security)
    plot_far_frr(security)
    plot_photometric(stress)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
