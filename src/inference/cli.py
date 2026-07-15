"""CLI JSON para comprobar el modelo y ejecutar verificación facial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.inference.config import load_model_config
from src.inference.errors import InferencePackageError
from src.inference.face_verifier import FaceVerifier
from src.inference.model_loader import check_model_artifact

DEFAULT_CONFIG = Path("config/model_config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inferencia del modelo facial final")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-model", help="Valida configuración y artefacto Keras")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    pair = subparsers.add_parser("verify-pair", help="Compara referencia y captura")
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--capture", type=Path, required=True)
    pair.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    references = subparsers.add_parser(
        "verify-references", help="Compara una captura contra varias referencias"
    )
    references.add_argument("--capture", type=Path, required=True)
    references.add_argument("--references", type=Path, nargs="+", required=True)
    references.add_argument("--strategy", choices=("max", "mean"), default="max")
    references.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.command == "check-model":
        return check_model_artifact(load_model_config(args.config))
    verifier = FaceVerifier.from_config(args.config)
    if args.command == "verify-pair":
        return verifier.verify_pair(args.reference, args.capture).to_dict()
    return verifier.verify_against_references(
        args.capture, args.references, strategy=args.strategy
    ).to_dict()


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = run(args)
    except (InferencePackageError, ValueError) as error:
        print(json.dumps({"status": "ERROR", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
