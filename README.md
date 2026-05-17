# Sistema de Control de Acceso Dual

Proyecto universitario del curso de Deep Learning. Consiste en un prototipo de control de acceso de doble factor que combina identificación por RFID con verificación facial mediante una **Red Neuronal Siamesa** con aprendizaje One-Shot/Few-Shot. A diferencia de un clasificador facial tradicional, la Red Siamesa no predice identidades directamente: aprende a medir la similitud entre pares de imágenes, lo que permite verificar usuarios nuevos sin reentrenar el modelo. El sistema corre en la laptop con Python y admite verificación facial multi-vista usando una sola cámara web.

## Funcionamiento general

1. El usuario presenta su tarjeta RFID. En la primera fase, este proceso será simulado; en una fase posterior, se usará un lector RFID real conectado al ESP32.
2. Si el UID del RFID es reconocido, la laptop captura una imagen del rostro mediante la cámara web.
3. La Red Siamesa compara el rostro capturado contra todas las imágenes de referencia disponibles para ese usuario en el `support_set` y utiliza la puntuación de similitud máxima.
4. La laptop determina si el acceso es concedido o denegado.
5. En la segunda fase, la laptop enviará `GRANTED` o `DENIED` al ESP32 mediante comunicación serial USB.
6. El ESP32 activará un servomotor si el acceso es concedido o un buzzer si el acceso es denegado.

## Verificación facial multi-vista

El sistema utiliza una única cámara web para capturar el rostro del usuario. No se requieren múltiples cámaras.

### Vistas soportadas

| Vista | Descripción |
|-------|-------------|
| `frontal` | Rostro de frente |
| `left` | Perfil o giro hacia la izquierda |
| `right` | Perfil o giro hacia la derecha |
| `mixed` | Combinación de vistas en el conjunto de entrenamiento |

### Support set por usuario

Cada usuario autorizado puede tener hasta tres imágenes de referencia en su carpeta dentro de `data/support_set/`:

```text
data/support_set/<nombre_usuario>/
├── frontal.jpg
├── left.jpg
└── right.jpg
```

No es obligatorio contar con las tres vistas; el sistema opera con las referencias disponibles.

### Estrategia de inferencia

Durante la verificación, la Red Siamesa compara el rostro capturado contra **todas** las referencias disponibles para el usuario asociado al UID del RFID. El acceso se concede o deniega según la **puntuación de similitud máxima** obtenida entre el rostro capturado y cualquiera de las referencias.

## Fases del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Laptop + cámara web + modelo siamesa + RFID simulado + verificación multi-vista | En progreso |
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
│   └── support_set/    # Imágenes de referencia por usuario (frontal, left, right); privado, no se sube
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

- Recolectar imágenes o videos faciales del equipo en vistas frontal, lateral izquierda, lateral derecha y mixta.
- Preprocesar las imágenes a una resolución de 112×112 píxeles.
- Generar pares positivos (misma persona, distintas vistas) y negativos (personas diferentes) para entrenar la Red Siamesa.
- Definir y entrenar el modelo de Deep Learning.
- Simular el UID de una tarjeta RFID.
- Capturar el rostro en tiempo real usando la cámara web.
- Comparar el rostro capturado contra todas las referencias multi-vista del usuario y usar la puntuación máxima.
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
