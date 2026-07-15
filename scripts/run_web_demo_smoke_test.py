"""Smoke test HTTP reproducible para la demo web local.

El script no crea imágenes ni usa rutas fijas. Las referencias indicadas se envían
a la aplicación y quedan en el almacenamiento privado local de la demo.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from uuid import uuid4


def encode_multipart(
    fields: list[tuple[str, str]], files: list[tuple[str, Path]]
) -> tuple[bytes, str]:
    boundary = f"----dual-access-smoke-{uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class DemoClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def get(self, path: str) -> tuple[int, str]:
        return self._send(Request(urljoin(self.base_url, path.lstrip("/"))))

    def post_multipart(
        self,
        path: str,
        fields: list[tuple[str, str]],
        files: list[tuple[str, Path]],
    ) -> tuple[int, str]:
        body, content_type = encode_multipart(fields, files)
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        return self._send(request)

    def _send(self, request: Request) -> tuple[int, str]:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            return error.code, body


def normalized_page(body: str) -> str:
    return " ".join(html.unescape(body).split())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_page(status: int, body: str, *markers: str) -> str:
    page = normalized_page(body)
    require(status == 200, f"HTTP inesperado: {status}")
    for marker in markers:
        require(marker in page, f"No se encontró '{marker}' en la respuesta")
    return page


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"No existe el archivo: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--uid", default="01020304")
    parser.add_argument("--unknown-uid", default="99999999")
    parser.add_argument("--name", default="Usuario Demo 1")
    parser.add_argument("--reference", action="append", type=existing_file)
    parser.add_argument("--positive-capture", type=existing_file)
    parser.add_argument("--negative-capture", type=existing_file)
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def run(args: argparse.Namespace) -> None:
    client = DemoClient(args.base_url, args.timeout)

    status, body = client.get("/health")
    require(status == 200, f"/health respondió HTTP {status}")
    health = json.loads(body)
    expected_health = {
        "status": "OK",
        "model_config_found": True,
        "database_available": True,
    }
    for field, expected in expected_health.items():
        require(health.get(field) == expected, f"/health: {field}={health.get(field)!r}")
    require(bool(health.get("model_name")), "/health no informó model_name")
    require(isinstance(health.get("threshold"), (int, float)), "/health no informó threshold")
    print(
        "[OK] health: "
        f"modelo={health['model_name']} threshold={health['threshold']}"
    )

    if args.health_only:
        return

    missing = []
    if not args.reference:
        missing.append("--reference")
    if not args.positive_capture:
        missing.append("--positive-capture")
    if not args.negative_capture:
        missing.append("--negative-capture")
    require(not missing, f"Faltan argumentos para el flujo completo: {', '.join(missing)}")
    require(args.uid.upper() != args.unknown_uid.upper(), "El UID desconocido debe ser diferente")

    status, body = client.post_multipart(
        "/users",
        [("full_name", args.name), ("rfid_uid", args.uid)],
        [("references", path) for path in args.reference],
    )
    require_page(status, body, args.name, args.uid.upper(), "registrado")
    print(f"[OK] registro: uid={args.uid.upper()} referencias={len(args.reference)}")

    status, body = client.post_multipart(
        "/verify",
        [("rfid_uid", args.uid)],
        [("capture", args.positive_capture)],
    )
    require_page(status, body, "GRANTED", "RFID_AND_FACE_OK", args.name)
    print("[OK] verificación positiva: GRANTED")

    status, body = client.post_multipart(
        "/verify",
        [("rfid_uid", args.unknown_uid)],
        [("capture", args.negative_capture)],
    )
    require_page(status, body, "DENIED", "RFID_UNKNOWN", "No calculado")
    print("[OK] UID desconocido: DENIED / RFID_UNKNOWN")

    status, body = client.post_multipart(
        "/verify",
        [("rfid_uid", args.uid)],
        [("capture", args.negative_capture)],
    )
    require_page(status, body, "DENIED", "FACE_NO_MATCH", args.name)
    print("[OK] rostro incorrecto: DENIED / FACE_NO_MATCH")

    status, body = client.get("/history")
    require_page(
        status,
        body,
        args.uid.upper(),
        args.unknown_uid.upper(),
        "GRANTED",
        "RFID_UNKNOWN",
        "FACE_NO_MATCH",
    )
    print("[OK] historial: contiene los tres intentos del smoke test")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (AssertionError, json.JSONDecodeError, OSError, URLError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print("Smoke test completado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
