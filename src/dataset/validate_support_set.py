"""
Valida la estructura del support set de referencias faciales sin leer contenido de imágenes.

Estructura esperada:
    data/support_set/<person_id>/frontal.jpg
    data/support_set/<person_id>/left.jpg
    data/support_set/<person_id>/right.jpg

Códigos de salida:
    0 — support set vacío o todos los usuarios tienen referencias completas
    1 — faltan referencias requeridas, hay archivos no soportados,
        o (en modo --strict) el support set está vacío
"""

import argparse
import sys
from pathlib import Path

from src.config import SUPPORT_REFERENCE_VIEWS, SUPPORT_SET_DIR

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida la estructura del support set de referencias faciales."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retorna código de salida 1 si el support set está vacío.",
    )
    return parser.parse_args()


def find_user_directories() -> tuple[list[Path], list[str]]:
    """
    Retorna (user_dirs, unexpected_files).
    unexpected_files son entradas directamente en SUPPORT_SET_DIR que no son carpetas de usuario.
    """
    user_dirs = []
    unexpected_files = []
    for entry in sorted(SUPPORT_SET_DIR.iterdir()):
        if entry.is_dir():
            user_dirs.append(entry)
        else:
            unexpected_files.append(entry.name)
    return user_dirs, unexpected_files


def find_reference_for_view(user_dir: Path, view: str) -> str | None:
    """Busca el archivo de referencia para una vista dada; retorna el nombre si existe, None si no."""
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = user_dir / f"{view}{ext}"
        if candidate.is_file():
            return candidate.name
    return None


def inspect_user_support_set(user_dir: Path) -> dict:
    """Retorna un dict con found, missing y unsupported para un directorio de usuario."""
    found = {}
    missing = []

    for view in SUPPORT_REFERENCE_VIEWS:
        ref = find_reference_for_view(user_dir, view)
        if ref:
            found[view] = ref
        else:
            missing.append(view)

    # Nombres de archivo válidos: cualquier combinación de vista requerida + extensión soportada
    valid_names = {
        f"{view}{ext}"
        for view in SUPPORT_REFERENCE_VIEWS
        for ext in SUPPORTED_IMAGE_EXTENSIONS
    }
    unsupported = [
        entry.name
        for entry in sorted(user_dir.iterdir())
        if entry.is_file() and entry.name not in valid_names
    ]

    return {"found": found, "missing": missing, "unsupported": unsupported}


def print_expected_structure() -> None:
    print("Estructura esperada:")
    print("  data/support_set/")
    print("  └── <person_id>/")
    for view in SUPPORT_REFERENCE_VIEWS:
        print(f"        {view}.jpg")
    print()
    print(f"Extensiones soportadas: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}")


def validate_support_set(strict: bool) -> int:
    print("=== Validación del Support Set ===\n")

    if not SUPPORT_SET_DIR.is_dir():
        print(f"ERROR: Directorio de support set no encontrado: {SUPPORT_SET_DIR}")
        print()
        print_expected_structure()
        return 1

    user_dirs, unexpected_root_files = find_user_directories()

    if not user_dirs and not unexpected_root_files:
        print("El support set está vacío.")
        print()
        print_expected_structure()
        return 1 if strict else 0

    has_issues = bool(unexpected_root_files)

    if unexpected_root_files:
        print("Archivos inesperados directamente en data/support_set (se esperan solo carpetas de usuario):")
        for name in unexpected_root_files:
            print(f"  [INESPERADO] {name}")
        print()

    total_users = len(user_dirs)
    complete_users = 0
    incomplete_users = 0
    total_missing = 0

    for user_dir in user_dirs:
        result = inspect_user_support_set(user_dir)
        found = result["found"]
        missing = result["missing"]
        unsupported = result["unsupported"]

        user_ok = not missing and not unsupported
        if user_ok:
            complete_users += 1
        else:
            incomplete_users += 1
            has_issues = True

        total_missing += len(missing)

        print(f"  Usuario: {user_dir.name}")

        if found:
            print("    Referencias encontradas:")
            for view, filename in found.items():
                print(f"      [OK] {view}: {filename}")
        else:
            print("    Referencias encontradas: ninguna")

        if missing:
            print("    Referencias faltantes:")
            for view in missing:
                print(f"      [FALTANTE] {view}")
        else:
            print("    Referencias faltantes: ninguna")

        if unsupported:
            print("    Archivos no soportados:")
            for name in unsupported:
                print(f"      [NO SOPORTADO] {name}")
        else:
            print("    Archivos no soportados: ninguno")

        print()

    print("=== Resumen ===")
    print(f"  Usuarios encontrados  : {total_users}")
    print(f"  Usuarios completos    : {complete_users}")
    print(f"  Usuarios incompletos  : {incomplete_users}")
    print(f"  Referencias faltantes : {total_missing}")

    return 1 if has_issues else 0


def main() -> None:
    args = parse_args()
    exit_code = validate_support_set(strict=args.strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
