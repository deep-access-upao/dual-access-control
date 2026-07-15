"""Calibra umbrales de seguridad con validation y evalúa splits separados."""

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

from src.config import EXPERIMENTS_DIR, PROJECT_ROOT, SAVED_MODEL_DIR
from src.evaluation.evaluate import compute_metrics, threshold_sweep
from src.utils.tensorflow_runtime import SUPPORTED_DEVICES, configure_tensorflow_runtime


DEFAULT_EXPERIMENT = "baseline_formal/baseline_con_aumento"
CALIBRATION_DIR_NAME = "security_threshold_calibration"
FAR_LIMITS = {
    "far_lte_1": 0.01,
    "far_lte_2": 0.02,
    "far_lte_3": 0.03,
}
CRITERIA = (
    "max_f1",
    "balanced_far_frr",
    "far_lte_1",
    "far_lte_2",
    "far_lte_3",
    "security_first",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara umbrales seleccionados exclusivamente con validation y los "
            "evalúa en test limpio y stress."
        )
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICES,
        default="auto",
        help="Dispositivo para generar scores de stress cuando no existe caché.",
    )
    parser.add_argument(
        "--refresh-stress-predictions",
        action="store_true",
        help="Regenera los scores de stress con el modelo local.",
    )
    return parser.parse_args()


def load_predictions(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Faltan las predicciones históricas de {split}: {path}")
    frame = pd.read_csv(path)
    required = {"y_true", "y_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}")
    result = frame[["y_true", "y_score"]].copy()
    result["y_true"] = result["y_true"].astype(int)
    result["y_score"] = result["y_score"].astype(float)
    if result.empty:
        raise ValueError(f"Las predicciones de {split} están vacías: {path}")
    if not set(result["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: y_true debe contener solo 0 y 1")
    if not np.isfinite(result["y_score"]).all():
        raise ValueError(f"{path}: y_score contiene valores no finitos")
    return result


def build_security_search(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    search = threshold_sweep(y_true, y_score).copy()
    search["far_frr_gap"] = (search["far"] - search["frr"]).abs()
    return search


def _best_f1(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["f1", "far", "frr", "threshold"],
        ascending=[False, True, True, False],
        kind="stable",
    ).iloc[0]


def select_security_candidate(search: pd.DataFrame, criterion: str) -> tuple[pd.Series | None, str]:
    """Selecciona un candidato sin recibir ni consultar datos de test."""
    if criterion == "max_f1":
        return _best_f1(search), "Máximo F1 en validation; desempate por menor FAR y FRR."

    if criterion == "balanced_far_frr":
        selected = search.sort_values(
            ["far_frr_gap", "f1", "far", "threshold"],
            ascending=[True, False, True, False],
            kind="stable",
        ).iloc[0]
        return selected, "Mínima diferencia absoluta entre FAR y FRR en validation."

    if criterion in FAR_LIMITS:
        limit = FAR_LIMITS[criterion]
        feasible = search[search["far"] <= limit + 1e-12]
        if feasible.empty:
            return None, f"No existe threshold con FAR <= {limit:.0%} en validation."
        return _best_f1(feasible), (
            f"Mejor F1 entre thresholds con FAR <= {limit:.0%} en validation."
        )

    if criterion == "security_first":
        # Límite operativo: prioriza FAR dentro de 2%, pero evita degradar la
        # experiencia legítima por encima de 5% de FRR en validation.
        primary = search[
            (search["far"] <= 0.02 + 1e-12)
            & (search["frr"] <= 0.05 + 1e-12)
        ]
        if not primary.empty:
            selected = primary.sort_values(
                ["far", "f1", "frr", "threshold"],
                ascending=[True, False, True, False],
                kind="stable",
            ).iloc[0]
            return selected, (
                "Menor FAR con FAR <= 2% y FRR <= 5% en validation; "
                "desempate por mayor F1."
            )

        fallback = search[search["far"] <= 0.02 + 1e-12]
        if not fallback.empty:
            return _best_f1(fallback), (
                "Fallback: mejor F1 con FAR <= 2%; no hubo candidato que además "
                "mantuviera FRR <= 5%."
            )

        fallback = search[search["far"] <= 0.03 + 1e-12]
        if not fallback.empty:
            return _best_f1(fallback), (
                "Fallback: mejor F1 con FAR <= 3%; no hubo candidato con FAR <= 2%."
            )

        selected = search.sort_values(
            ["far", "frr", "f1", "threshold"],
            ascending=[True, True, False, False],
            kind="stable",
        ).iloc[0]
        return selected, "Fallback final: menor FAR disponible en validation."

    raise ValueError(f"Criterio no soportado: {criterion}")


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    confusion = metrics["confusion_matrix"]
    row = {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_f1": metrics["f1"],
        f"{prefix}_far": metrics["far"],
        f"{prefix}_frr": metrics["frr"],
        f"{prefix}_tn": confusion["tn"],
        f"{prefix}_fp": confusion["fp"],
        f"{prefix}_fn": confusion["fn"],
        f"{prefix}_tp": confusion["tp"],
    }
    if prefix == "test":
        row["test_roc_auc"] = metrics["roc_auc"]
    return row


def build_candidate_table(
    search: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    unavailable: list[dict] = []
    validation_labels = validation["y_true"].to_numpy()
    validation_scores = validation["y_score"].to_numpy()
    test_labels = test["y_true"].to_numpy()
    test_scores = test["y_score"].to_numpy()

    for criterion in CRITERIA:
        selected, comment = select_security_candidate(search, criterion)
        if selected is None:
            unavailable.append({"criterion": criterion, "comment": comment})
            continue
        threshold = float(selected["threshold"])
        validation_metrics = compute_metrics(
            validation_labels, validation_scores, threshold
        )
        test_metrics = compute_metrics(test_labels, test_scores, threshold)
        rows.append(
            {
                "criterion": criterion,
                "threshold": threshold,
                **flatten_metrics("validation", validation_metrics),
                **flatten_metrics("test", test_metrics),
                "comment": comment,
            }
        )
    return pd.DataFrame(rows), unavailable


def load_or_collect_stress_predictions(
    output_dir: Path,
    model_path: Path,
    batch_size: int,
    seed: int,
    device: str,
    refresh: bool,
) -> tuple[pd.DataFrame, str, str]:
    cache_path = output_dir / "stress_predictions.csv"
    metadata_path = output_dir / "stress_predictions_metadata.json"
    if cache_path.is_file() and not refresh:
        frame = pd.read_csv(cache_path)
        required = {"condition", "y_true", "y_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{cache_path}: faltan columnas {sorted(missing)}")
        selected_device = "unknown"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                selected_device = str(json.load(file).get("evaluation_device", "unknown"))
        else:
            previous_summary = output_dir / "security_threshold_calibration.json"
            if previous_summary.is_file():
                with previous_summary.open(encoding="utf-8") as file:
                    selected_device = str(
                        json.load(file).get("evaluation_device", "unknown")
                    )
            save_json(
                {
                    "evaluation_device": selected_device,
                    "stress_seed": seed,
                    "source": "generated_with_model_then_cached",
                },
                metadata_path,
            )
        return frame, selected_device, "cached"

    from src.dataset.dataloader import TEST_CSV
    from src.evaluation.evaluate import load_model
    from src.evaluation.evaluate_stress import collect_predictions, create_stress_dataset
    from src.evaluation.stress_tests import STRESS_CONDITIONS

    if not TEST_CSV.is_file():
        raise FileNotFoundError(
            f"Falta {TEST_CSV}; configura DUAL_ACCESS_DATA_DIR con los datos locales."
        )
    runtime = configure_tensorflow_runtime(device)
    model = load_model(model_path)
    frames = []
    for condition in STRESS_CONDITIONS:
        print(f"Generando scores de stress: {condition}")
        dataset = create_stress_dataset(TEST_CSV, condition, batch_size, seed)
        y_true, y_score = collect_predictions(model, dataset)
        frames.append(
            pd.DataFrame(
                {"condition": condition, "y_true": y_true, "y_score": y_score}
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(cache_path, index=False)
    save_json(
        {
            "evaluation_device": runtime.selected_device,
            "stress_seed": seed,
            "source": "generated_with_model_then_cached",
        },
        metadata_path,
    )
    return frame, runtime.selected_device, "generated"


def evaluate_stress_candidates(
    candidates: pd.DataFrame, stress_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    worst_by_criterion: dict[str, dict] = {}
    for candidate in candidates.itertuples(index=False):
        criterion_rows = []
        for condition, condition_frame in stress_predictions.groupby(
            "condition", sort=False
        ):
            metrics = compute_metrics(
                condition_frame["y_true"].to_numpy(),
                condition_frame["y_score"].to_numpy(),
                float(candidate.threshold),
            )
            row = {
                "criterion": candidate.criterion,
                "threshold": float(candidate.threshold),
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
            rows.append(row)
            criterion_rows.append(row)
        worst = min(criterion_rows, key=lambda row: (row["accuracy"], row["f1"]))
        worst_by_criterion[candidate.criterion] = worst
    return pd.DataFrame(rows), worst_by_criterion


def add_stress_summary(
    candidates: pd.DataFrame, worst_by_criterion: dict[str, dict]
) -> pd.DataFrame:
    result = candidates.copy()
    result["worst_stress"] = result["criterion"].map(
        lambda criterion: worst_by_criterion[criterion]["condition"]
    )
    for metric in ("accuracy", "f1", "far", "frr"):
        result[f"worst_stress_{metric}"] = result["criterion"].map(
            lambda criterion, key=metric: worst_by_criterion[criterion][key]
        )
    return result


def plot_threshold_curves(search: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(search["threshold"], search["far"], label="FAR")
    ax.plot(search["threshold"], search["frr"], label="FRR")
    ax.set(xlabel="Threshold", ylabel="Tasa", title="FAR vs FRR en validation")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_far_vs_frr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(search["threshold"], search["f1"], color="tab:green")
    ax.set(xlabel="Threshold", ylabel="F1", title="F1 vs threshold en validation")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_f1_vs_threshold.png", dpi=130)
    plt.close(fig)


def plot_candidate_comparisons(candidates: pd.DataFrame, output_dir: Path) -> None:
    labels = candidates["criterion"].str.replace("_", "\n")
    positions = np.arange(len(candidates))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric, display in (
        (-1.5, "test_accuracy", "Accuracy"),
        (-0.5, "test_f1", "F1"),
        (0.5, "test_far", "FAR"),
        (1.5, "test_frr", "FRR"),
    ):
        ax.bar(positions + offset * width, candidates[metric], width, label=display)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Candidatos evaluados en test limpio")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "test_candidate_comparison.png", dpi=130)
    plt.close(fig)

    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(positions - width / 2, candidates["validation_far"], width, label="FAR")
    ax.bar(positions + width / 2, candidates["validation_frr"], width, label="FRR")
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("FAR/FRR de candidatos en validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_candidate_far_frr.png", dpi=130)
    plt.close(fig)


def plot_score_distributions(
    validation: pd.DataFrame, candidates: pd.DataFrame, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    negatives = validation.loc[validation["y_true"] == 0, "y_score"]
    positives = validation.loc[validation["y_true"] == 1, "y_score"]
    ax.hist(negatives, bins=35, alpha=0.55, label="Impostores (0)")
    ax.hist(positives, bins=35, alpha=0.55, label="Genuinos (1)")
    threshold_labels: dict[float, list[str]] = {}
    for row in candidates.itertuples(index=False):
        threshold = float(row.threshold)
        threshold_labels.setdefault(threshold, []).append(row.criterion)
    for threshold, labels in threshold_labels.items():
        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=1,
            label=" / ".join(labels),
        )
    ax.set(
        xlabel="Score de similitud",
        ylabel="Frecuencia",
        title="Distribución de scores en validation",
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_score_distributions.png", dpi=130)
    plt.close(fig)


def save_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def build_recommendations(candidates: pd.DataFrame) -> dict[str, dict]:
    by_criterion = candidates.set_index("criterion")

    def recommendation(criterion: str, rationale: str) -> dict:
        row = by_criterion.loc[criterion]
        return {
            "criterion": criterion,
            "threshold": float(row["threshold"]),
            "rationale": rationale,
        }

    return {
        "maximum_f1": recommendation(
            "max_f1", "Referencia histórica que maximiza F1 en validation."
        ),
        "moderate_security": recommendation(
            "security_first",
            "Limita FAR a 2% y FRR a 5% en validation antes de desempatar.",
        ),
        "strict_security": recommendation(
            "far_lte_1",
            "Impone FAR <= 1% en validation; debe revisarse el aumento de FRR.",
        ),
        "rfid_face_demo": recommendation(
            "security_first",
            "Compromiso defendible para la demo entre intrusiones y rechazos legítimos.",
        ),
        "less_strict_demo": recommendation(
            "balanced_far_frr",
            "Alternativa menos estricta que aproxima FAR y FRR en validation.",
        ),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = EXPERIMENTS_DIR / args.experiment_name
    model_path = args.model_path or SAVED_MODEL_DIR / f"{args.experiment_name}.keras"
    model_path = Path(model_path)
    if not model_path.is_file():
        print(f"ERROR: falta el modelo requerido: {model_path}")
        sys.exit(1)
    if not experiment_dir.is_dir():
        print(f"ERROR: faltan los resultados del experimento: {experiment_dir}")
        sys.exit(1)

    output_dir = experiment_dir / CALIBRATION_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        validation = load_predictions(
            experiment_dir / "validation_predictions.csv", "validation"
        )
        test = load_predictions(experiment_dir / "test_predictions.csv", "test limpio")
        search = build_security_search(
            validation["y_true"].to_numpy(), validation["y_score"].to_numpy()
        )
        candidates, unavailable = build_candidate_table(search, validation, test)
        stress_predictions, selected_device, stress_prediction_source = (
            load_or_collect_stress_predictions(
                output_dir,
                model_path,
                args.batch_size,
                args.seed,
                args.device,
                args.refresh_stress_predictions,
            )
        )
        stress_results, worst = evaluate_stress_candidates(
            candidates, stress_predictions
        )
        candidates = add_stress_summary(candidates, worst)

        search.to_csv(output_dir / "validation_threshold_search.csv", index=False)
        candidates.to_csv(output_dir / "security_threshold_comparison.csv", index=False)
        stress_results.to_csv(
            output_dir / "stress_metrics_by_candidate.csv", index=False
        )
        plot_threshold_curves(search, output_dir)
        plot_candidate_comparisons(candidates, output_dir)
        plot_score_distributions(validation, candidates, output_dir)

        recommendations = build_recommendations(candidates)
        payload = {
            "experiment_name": args.experiment_name,
            "model_path": str(model_path.resolve().relative_to(PROJECT_ROOT)),
            "calibration_split": "validation",
            "test_used_for_selection": False,
            "stress_used_for_selection": False,
            "stress_seed": args.seed,
            "evaluation_device": selected_device,
            "stress_prediction_source": stress_prediction_source,
            "security_first_policy": (
                "Menor FAR entre thresholds con FAR <= 2% y FRR <= 5% en validation; "
                "desempate por mayor F1. Si no existe, mejor F1 con FAR <= 2%; "
                "después FAR <= 3%; finalmente menor FAR disponible."
            ),
            "unavailable_criteria": unavailable,
            "recommendations": recommendations,
            "candidates": candidates.to_dict(orient="records"),
        }
        save_json(payload, output_dir / "security_threshold_calibration.json")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print("\n=== Calibración de umbrales de seguridad ===")
    print("Selección: exclusivamente validation")
    print("Test limpio y stress: solo evaluación")
    print(f"Dispositivo de evaluación: {selected_device}")
    display_columns = [
        "criterion",
        "threshold",
        "validation_far",
        "validation_frr",
        "validation_f1",
        "test_far",
        "test_frr",
        "test_f1",
        "test_accuracy",
        "test_fp",
        "test_fn",
        "worst_stress",
    ]
    print(candidates[display_columns].to_string(index=False))
    print(f"\nResultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
