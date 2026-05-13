# Sistema de Control de Acceso Dual

Proyecto universitario del curso de Deep Learning. Consiste en un prototipo de control de acceso de doble factor que combina identificación por RFID con verificación facial mediante una Red Neuronal Siamesa y One-Shot Learning.

## Funcionamiento general

1. El usuario presenta su tarjeta RFID. En la primera fase, este proceso será simulado; en una fase posterior, se usará un lector RFID real conectado al ESP32.
2. Si el UID del RFID es reconocido, la laptop captura una imagen del rostro mediante la cámara web.
3. La Red Siamesa compara el rostro capturado con una imagen de referencia almacenada en el `support_set`.
4. La laptop determina si el acceso es concedido o denegado.
5. En la segunda fase, la laptop enviará `GRANTED` o `DENIED` al ESP32 mediante comunicación serial USB.
6. El ESP32 activará un servomotor si el acceso es concedido o un buzzer si el acceso es denegado.

## Fases del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Laptop + cámara web + modelo siamesa + RFID simulado | En progreso |
| 2 | Laptop + cámara web + modelo + ESP32 con servomotor/buzzer | Pendiente |
| 3 | ESP32 lee RFID real y envía el UID a la laptop | Pendiente |
| 4 | Conversión opcional a TensorFlow Lite o despliegue embebido | Futuro |

## Tecnologías

| Capa | Herramienta |
|------|-------------|
| Deep Learning | TensorFlow / Keras |
| Visión por computadora | OpenCV |
| Manejo de datos | NumPy, Pandas |
| Visualización | Matplotlib |
| Evaluación del modelo | scikit-learn |
| Comunicación serial | pyserial |
| Microcontrolador | ESP32 con Arduino IDE |
| Lenguaje principal | Python 3.10+ |

## Estructura del repositorio

```text
dual-access-control/
├── data/
│   ├── raw/            # Videos e imágenes originales (privados, no se suben al repositorio)
│   ├── processed/      # Rostros recortados y redimensionados (privados, no se suben)
│   ├── pairs/          # Pares positivos y negativos para entrenamiento (privados, no se suben)
│   └── support_set/    # Una imagen de referencia por usuario autorizado (privado, no se sube)
├── docs/               # Documentación, notas técnicas y diagramas del proyecto
├── esp32/
│   ├── servo_buzzer_controller/  # Código Arduino para controlar servomotor y buzzer
│   └── rfid_reader/              # Código Arduino para lectura RFID en la fase 3
├── models/
│   ├── saved_model/    # Modelo entrenado en formato Keras/TensorFlow (no se sube)
│   └── tflite/         # Modelo convertido a TensorFlow Lite (no se sube)
├── notebooks/          # Notebooks de exploración, entrenamiento o visualización
├── outputs/
│   ├── logs/           # Registros de entrenamiento o ejecución
│   ├── metrics/        # Resultados de evaluación del modelo
│   └── plots/          # Gráficas de pérdida, accuracy, ROC, etc.
├── src/
│   ├── dataset/        # Generación de pares y carga de datos
│   ├── evaluation/     # Scripts para evaluación del modelo
│   ├── hardware/       # Comunicación serial entre laptop y ESP32
│   ├── inference/      # Inferencia en tiempo real con cámara web
│   ├── models/         # Definición de la arquitectura de la Red Siamesa
│   ├── preprocessing/  # Detección, recorte y preprocesamiento facial
│   └── training/       # Scripts de entrenamiento
├── .gitignore
├── README.md
└── requirements.txt
```

## Instalación del entorno

```bash
# Crear entorno virtual
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Alcance actual: Fase 1

En la primera fase, el sistema funcionará completamente desde la laptop, sin depender del hardware físico.

Objetivos de esta fase:

- Recolectar videos o imágenes faciales del equipo.
- Preprocesar las imágenes a una resolución de 112×112 píxeles.
- Generar pares positivos y negativos para entrenar la Red Siamesa.
- Definir y entrenar el modelo de Deep Learning.
- Simular el UID de una tarjeta RFID.
- Capturar el rostro en tiempo real usando la cámara web.
- Comparar el rostro capturado con la imagen de referencia del usuario.
- Mostrar en consola si el acceso fue concedido o denegado.

## Integración con hardware: Fase 2 y posteriores

En la segunda fase, el sistema incorporará el ESP32 como módulo físico de respuesta.

Flujo esperado:

```text
Laptop con Python
↓
Captura facial con cámara web
↓
Modelo de Red Siamesa decide acceso
↓
Laptop envía GRANTED o DENIED por USB serial
↓
ESP32 activa servomotor o buzzer
```

Conexiones previstas:

| Componente | Conexión prevista |
|-----------|-------------------|
| ESP32 | Conectado a la laptop por USB |
| Servomotor | GPIO 18 |
| Buzzer | GPIO 19 |
| RFID RC522 | Pines SPI del ESP32 en la fase 3 |

## Consideraciones sobre privacidad

El proyecto utiliza imágenes faciales reales de los integrantes o voluntarios. Por ello, los videos, imágenes procesadas, pares de entrenamiento, imágenes de referencia y modelos entrenados no deben subirse al repositorio.

Las carpetas relacionadas al dataset se mantienen en GitHub únicamente mediante archivos `.gitkeep`.
