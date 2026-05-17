from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET_DIR = DATA_DIR / "raw"
PROCESSED_DATASET_DIR = DATA_DIR / "processed"
PAIRS_DIR = DATA_DIR / "pairs"
SUPPORT_SET_DIR = DATA_DIR / "support_set"

# models/
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODEL_DIR = MODELS_DIR / "saved"
TFLITE_MODEL_DIR = MODELS_DIR / "tflite"

# outputs/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Configuración de imagen
# ---------------------------------------------------------------------------

IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# ---------------------------------------------------------------------------
# Vistas de cara
# ---------------------------------------------------------------------------

SUPPORTED_FACE_VIEWS = ["frontal", "left", "right", "mixed"]
SUPPORT_REFERENCE_VIEWS = ["frontal", "left", "right"]

# ---------------------------------------------------------------------------
# Configuración del modelo / inferencia
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD = 0.5
ACCESS_GRANTED = "GRANTED"
ACCESS_DENIED = "DENIED"

# ---------------------------------------------------------------------------
# Configuración serial (integración ESP32 — Fase 2+)
# ---------------------------------------------------------------------------

DEFAULT_SERIAL_PORT = "COM3"
DEFAULT_BAUD_RATE = 9600
SERIAL_TIMEOUT_SECONDS = 2

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Dual Access Control — Configuración ===\n")

    print("--- Rutas ---")
    print(f"PROJECT_ROOT          : {PROJECT_ROOT}")
    print(f"DATA_DIR              : {DATA_DIR}")
    print(f"RAW_DATASET_DIR       : {RAW_DATASET_DIR}")
    print(f"PROCESSED_DATASET_DIR : {PROCESSED_DATASET_DIR}")
    print(f"PAIRS_DIR             : {PAIRS_DIR}")
    print(f"SUPPORT_SET_DIR       : {SUPPORT_SET_DIR}")
    print(f"MODELS_DIR            : {MODELS_DIR}")
    print(f"SAVED_MODEL_DIR       : {SAVED_MODEL_DIR}")
    print(f"TFLITE_MODEL_DIR      : {TFLITE_MODEL_DIR}")
    print(f"OUTPUTS_DIR           : {OUTPUTS_DIR}")
    print(f"LOGS_DIR              : {LOGS_DIR}")
    print(f"METRICS_DIR           : {METRICS_DIR}")
    print(f"PLOTS_DIR             : {PLOTS_DIR}")

    print("\n--- Imagen ---")
    print(f"IMAGE_SIZE            : {IMAGE_SIZE}")
    print(f"INPUT_SHAPE           : {INPUT_SHAPE}")
    print(f"IMAGE_CHANNELS        : {IMAGE_CHANNELS}")

    print("\n--- Vistas de cara ---")
    print(f"SUPPORTED_FACE_VIEWS  : {SUPPORTED_FACE_VIEWS}")
    print(f"SUPPORT_REFERENCE_VIEWS: {SUPPORT_REFERENCE_VIEWS}")

    print("\n--- Inferencia ---")
    print(f"SIMILARITY_THRESHOLD  : {DEFAULT_SIMILARITY_THRESHOLD}")
    print(f"ACCESS_GRANTED        : {ACCESS_GRANTED}")
    print(f"ACCESS_DENIED         : {ACCESS_DENIED}")

    print("\n--- Serial (ESP32) ---")
    print(f"SERIAL_PORT           : {DEFAULT_SERIAL_PORT}")
    print(f"BAUD_RATE             : {DEFAULT_BAUD_RATE}")
    print(f"SERIAL_TIMEOUT        : {SERIAL_TIMEOUT_SECONDS}s")
