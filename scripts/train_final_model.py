"""Orquesta la regeneración reproducible del modelo final; no entrena por defecto.

Sin ``--execute`` solo imprime el plan. Para evitar sobrescrituras, una ejecución
con artefactos existentes exige además ``--overwrite``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS_DIR, SAVED_MODEL_DIR

EXPERIMENT = "baseline_formal/baseline_con_aumento"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regeneración controlada del modelo final")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--execute", action="store_true", help="Ejecuta el plan; sin este flag solo lo muestra")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Confirma que se permite sobrescribir artefactos existentes",
    )
    return parser.parse_args()


def build_commands(device: str) -> list[list[str]]:
    python = sys.executable
    common = ["--experiment-name", EXPERIMENT]
    return [
        [
            python,
            "-m",
            "src.training.train",
            *common,
            "--model-variant",
            "baseline",
            "--augmentation",
            "--seed",
            "42",
            "--epochs",
            "10",
            "--batch-size",
            "64",
            "--learning-rate",
            "0.0001",
            "--patience",
            "5",
            "--device",
            device,
        ],
        [python, "-m", "src.evaluation.evaluate", "--mode", "calibrate", *common, "--batch-size", "64", "--device", device],
        [python, "-m", "src.evaluation.evaluate", "--mode", "test", *common, "--batch-size", "64", "--device", device],
        [
            python,
            "-m",
            "src.evaluation.calibrate_security_thresholds",
            *common,
            "--batch-size",
            "64",
            "--seed",
            "2026",
            "--device",
            device,
            "--refresh-stress-predictions",
        ],
    ]


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def main() -> None:
    args = parse_args()
    commands = build_commands(args.device)
    print("Plan reproducible del modelo final (entrenamiento -> validation -> test -> security_first):")
    for index, command in enumerate(commands, start=1):
        print(f"{index}. {display_command(command)}")

    if not args.execute:
        print("\nSimulación: no se ejecutó ningún comando. Usa --execute únicamente en WSL/GPU.")
        return

    model_path = SAVED_MODEL_DIR / f"{EXPERIMENT}.keras"
    experiment_dir = EXPERIMENTS_DIR / EXPERIMENT
    if (model_path.exists() or experiment_dir.exists()) and not args.overwrite:
        raise SystemExit(
            "Ya existen artefactos del experimento. Revisa un respaldo y confirma con --overwrite."
        )
    for command in commands:
        subprocess.run(command, check=True)

    print(
        "\nRevisa security_threshold_calibration.json y confirma manualmente el candidato "
        "security_first antes de actualizar cualquier configuración de producción."
    )


if __name__ == "__main__":
    main()
