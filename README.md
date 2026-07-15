# Dual Access Control

Prototipo académico de control de acceso de doble factor. Combina un UID RFID simulado con verificación facial mediante una red siamesa CNN y registra la decisión en una demo web local construida con FastAPI y SQLite.

> Estado: demo local funcional. No es un sistema biométrico listo para producción.

## Problema y objetivo

Una credencial RFID puede perderse, prestarse o copiarse. El proyecto añade un segundo factor biométrico: el rostro capturado se compara con las referencias del usuario asociado al UID. Deep Learning es pertinente porque las variaciones de iluminación, pose, encuadre y apariencia no se resuelven de forma robusta con reglas manuales; una red siamesa aprende una función de similitud y permite incorporar referencias de un usuario sin convertir el problema en un clasificador cerrado de identidades.

El objetivo medible fue construir y evaluar un flujo reproducible capaz de verificar pares faciales y demostrar la regla de acceso completa. Los indicadores son accuracy, precision, recall, F1, FAR, FRR, ROC AUC y el funcionamiento integral de la demo web.

## Arquitectura general

```text
UID simulado en web
        |
        v
usuario conocido y activo? -- no --> DENIED + historial
        |
       sí
        v
captura facial -> preprocesamiento 112x112 RGB -> red siamesa
        |
        v
máximo score contra referencias del usuario
        |
        v
score >= 0.3128704727? -- no --> DENIED + historial
        |
       sí
        v
GRANTED + historial
```

Reglas finales:

- Facial: `score >= 0.3128704727 => MATCH`; de lo contrario, `NO_MATCH`.
- Acceso: `RFID conocido y usuario activo AND rostro verificado => GRANTED`; cualquier otro caso es `DENIED`.
- Fallos de configuración o inferencia se resuelven de forma segura como `DENIED`.

El RFID se simula mediante el ingreso del UID en la web. Es una decisión de alcance por tiempo: se valida la misma lógica que recibiría el UID desde un lector real, `UID -> usuario -> rostro -> decisión -> historial`, sin depender de ESP32 ni hardware físico.

## Dataset y protocolo sin fuga

El dataset privado contiene 11 personas y 44 videos. De 1,702 imágenes detectadas originalmente, 1,544 fueron utilizables y 158 se rechazaron por calidad. Las imágenes se decodifican en RGB, se redimensionan a `112x112x3`, se convierten a `float32` y se normalizan al intervalo `[0, 1]`.

| Split | Imágenes | Videos | Pares |
|---|---:|---:|---:|
| Train | 739 | 22 | 4,000 |
| Validation | 400 | 11 | 500 |
| Test | 405 | 11 | 500 |

Los pares están balanceados entre positivos —misma persona— y negativos —personas distintas—. El split se hace por video antes de generar pares. Separar pares aleatoriamente habría permitido que frames de una misma secuencia aparecieran en train y evaluación, inflando las métricas. La auditoría final confirmó:

- imágenes compartidas: 0;
- videos compartidos: 0;
- pares repetidos: 0;
- hashes cruzados: 0.

Los datos faciales, manifests privados y CSV de pares no se versionan. El protocolo completo está en [docs/dataset_protocol.md](docs/dataset_protocol.md).

## Modelo final y resultados

El modelo seleccionado es `baseline_formal/baseline_con_aumento`: cuatro bloques `Conv2D + BatchNormalization + MaxPooling`, `Flatten`, una proyección densa, dropout, embedding de 128 componentes, distancia L1 y salida sigmoide de similitud. Se entrenó con Adam, learning rate `0.0001`, batch size 64, máximo de 10 épocas, patience 5 y seed 42. Los aumentos realistas se aplicaron solo en train; validation y test permanecieron limpios.

La configuración operativa está versionada en [config/model_config.json](config/model_config.json). El archivo `.keras` es privado y no se sube al repositorio.

### Test limpio con threshold `security_first`

| Accuracy | Precision | Recall | F1 | FAR | FRR | ROC AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9660 | 0.9794 | 0.9520 | 0.9655 | 0.0200 | 0.0480 | 0.9838 |

Matriz de confusión: `TN=245`, `FP=5`, `FN=12`, `TP=238` sobre 500 pares.

El criterio `max_f1` alcanzó F1 0.9803, pero FAR 0.0360. El threshold final `0.3128704727`, calibrado exclusivamente con validation, reduce el FAR de test a 0.0200 con el costo explícito de aumentar el FRR a 0.0480. Esta prioridad es coherente con un control de acceso de doble factor.

El aumento produjo una mejora pequeña en test limpio frente al baseline sin aumento, pero una mejora marcada de robustez. También se evaluó una variante con `GlobalAveragePooling2D`, embeddings L2 y coseno: fue más compacta y mejor en test limpio, pero rechazó entre 82% y 100% de pares genuinos bajo las tres degradaciones fotométricas críticas. Por ello no se seleccionó.

### Principales resultados de estrés

| Condición | Accuracy | F1 | FAR | FRR |
|---|---:|---:|---:|---:|
| Poca luz | 0.782 | 0.7471 | 0.080 | 0.356 |
| Sobreexposición | 0.832 | 0.8019 | 0.016 | 0.320 |
| Bajo contraste | 0.816 | 0.7909 | 0.064 | 0.304 |
| Ruido | 0.918 | 0.9118 | 0.012 | 0.152 |
| Blur | 0.952 | 0.9508 | 0.024 | 0.072 |

Poca luz, sobreexposición y bajo contraste son los fallos principales. En una operación real pueden rechazar usuarios legítimos; la mitigación inmediata es ofrecer recaptura, luz frontal uniforme y varias referencias. El análisis completo está en [docs/security_threshold_calibration.md](docs/security_threshold_calibration.md) y [docs/siamese_model_improvement.md](docs/siamese_model_improvement.md).

## Instalación

Requisitos: Python 3.10 o superior. Desde la raíz del repositorio, en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Coloca el modelo privado en la ruta declarada en `config/model_config.json`:

```text
models/saved_model/baseline_formal/baseline_con_aumento.keras
```

## Uso

### Verificar el artefacto del modelo

```powershell
python -m src.inference.cli check-model --config config/model_config.json
```

### Inferencia por línea de comandos

```powershell
python -m src.inference.cli verify-pair `
  --reference "C:\ruta\referencia.jpg" `
  --capture "C:\ruta\captura.jpg" `
  --config config/model_config.json

python -m src.inference.cli verify-references `
  --capture "C:\ruta\captura.jpg" `
  --references "C:\ruta\frontal.jpg" "C:\ruta\left.jpg" "C:\ruta\right.jpg" `
  --strategy max `
  --config config/model_config.json
```

### Demo web

```powershell
python -m src.web.app
```

Abre `http://127.0.0.1:8000`. Registra un usuario con UID y referencias, simula el UID, aporta una captura y consulta el historial. La base y las imágenes quedan bajo `data/demo/` y están excluidas de Git. Consulta [docs/web_demo.md](docs/web_demo.md).

## Tests y prueba integral

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

Con la web en ejecución:

```powershell
python scripts/run_web_demo_smoke_test.py --health-only
```

El smoke test completo requiere imágenes locales autorizadas:

```powershell
python scripts/run_web_demo_smoke_test.py `
  --base-url http://127.0.0.1:8000 `
  --uid 01020304 `
  --name "Usuario Demo 1" `
  --reference "C:\ruta\user_001\ref.jpg" `
  --positive-capture "C:\ruta\user_001\probe.jpg" `
  --negative-capture "C:\ruta\user_002\probe.jpg"
```

Usa el [plan de pruebas integrales](docs/integrated_demo_test_plan.md) y el [checklist de evidencias](docs/demo_evidence_checklist.md). No publiques las imágenes usadas.

## Limitaciones

- Dataset pequeño: 11 personas, sin validación con usuarios externos.
- Los splits comparten identidades; separan videos. Falta un protocolo estable de identidades no vistas con una muestra mayor.
- Estrés sintético y sensibilidad alta a degradaciones fotométricas.
- No existe un conjunto `validation-stress` independiente.
- El modelo final es una CNN propia; no usa backbone preentrenado ni fine-tuning.
- RFID simulado; no se integró lector físico ni ESP32 en la demo final.
- Demo local sin autenticación, HTTPS ni despliegue público.
- No se ha validado a escala productiva, con ataques de presentación o requisitos regulatorios.

Próximos pasos realistas: más personas y sesiones, `validation-stress` separado, controles de calidad y CLAHE evaluados sin contaminar test, MobileNetV2 congelado seguido de fine-tuning selectivo, detección de vida, integración RFID física y validación externa.

## Ética y uso responsable

Los rostros son datos sensibles. Solo deben capturarse con consentimiento informado, acceso mínimo, propósito definido y política de retención. No se suben fotos, videos, datasets, support sets, bases SQLite ni modelos. El tamaño y composición del dataset pueden introducir sesgos; FAR y FRR implican riesgos distintos para personas distintas. Un falso positivo compromete seguridad y un falso negativo niega acceso a un usuario legítimo. Este prototipo es académico y requiere evaluación técnica, de privacidad, equidad y seguridad antes de cualquier uso real.

## Estructura del repositorio

```text
config/       configuración versionada del modelo final
data/         datos privados locales y marcadores .gitkeep
docs/         metodología, uso, resultados, pruebas y rúbrica
esp32/        prototipos de hardware fuera del alcance de la demo final
models/       artefactos privados ignorados por Git
notebooks/    preparación, evaluación y resumen de presentación
outputs/      métricas y gráficos livianos de experimentos
scripts/      smoke test y plan de regeneración controlada
src/          dataset, modelos, entrenamiento, evaluación, inferencia y web
tests/        pruebas unitarias e integrales
```

## Documentación principal

- [Metodología final](docs/final_methodology.md)
- [Guía final de uso](docs/final_usage_guide.md)
- [Notebook resumen](notebooks/99_project_summary.ipynb)
- [Cobertura de la rúbrica](docs/rubric_coverage.md)
- [Checklist de presentación](docs/final_presentation_checklist.md)
