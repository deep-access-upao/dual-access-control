# Guía final de instalación y ejecución

## 1. Requisitos y alcance

Ejecuta los comandos desde la raíz del repositorio en Windows nativo. Para inferencia y web no se necesita GPU. El modelo `.keras`, el dataset, las fotos y la base SQLite son recursos privados y deben permanecer fuera de Git.

Esta guía no requiere entrenar. La sección de WSL/GPU documenta una regeneración excepcional, pero no debe ejecutarse durante una demo ni para reproducir la presentación.

## 2. Crear y activar el entorno

En PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación, puede usarse el intérprete directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Preparar recursos privados

El archivo [config/model_config.json](../config/model_config.json) espera:

```text
models/saved_model/baseline_formal/baseline_con_aumento.keras
```

No cambies el threshold para adaptar una demostración. El valor operativo es `0.3128704727` y la regla es `score >= threshold`.

Para la web, las referencias se cargan desde la interfaz y se copian a `data/demo/reference_images/`. SQLite se crea en `data/demo/dual_access_demo.sqlite3`. Ambos están ignorados por Git.

## 4. Verificar configuración y modelo

```powershell
python -m src.inference.cli check-model --config config/model_config.json
```

El comando debe devolver JSON con `status: OK`, nombre del modelo, threshold, forma de entrada y salida. Si informa que el modelo no existe, colócalo en la ruta configurada; no lo descargues ni lo subas al repositorio.

## 5. Inferencia CLI

### Comparar un par

```powershell
python -m src.inference.cli verify-pair `
  --reference "C:\ruta\autorizada\referencia.jpg" `
  --capture "C:\ruta\autorizada\captura.jpg" `
  --config config/model_config.json
```

La salida contiene score, threshold, `MATCH`/`NO_MATCH` y metadatos mínimos.

### Comparar contra varias referencias

```powershell
python -m src.inference.cli verify-references `
  --capture "C:\ruta\autorizada\captura.jpg" `
  --references `
    "C:\ruta\autorizada\frontal.jpg" `
    "C:\ruta\autorizada\left.jpg" `
    "C:\ruta\autorizada\right.jpg" `
  --strategy max `
  --config config/model_config.json
```

La demo usa `max`: basta que la captura se parezca suficientemente a una referencia válida. `mean` está disponible para análisis, pero no es la estrategia operativa documentada.

## 6. Ejecutar la web

```powershell
python -m src.web.app
```

Abre `http://127.0.0.1:8000`. El flujo recomendado es:

1. Registrar `Usuario Demo 1` con UID simulado `01020304` y referencias consentidas.
2. Confirmar que el usuario esté activo.
3. Ingresar el UID y cargar o capturar el rostro.
4. Revisar decisión, motivo, score y threshold.
5. Abrir el historial.

Comprobación de salud:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json
```

La web es local y no tiene autenticación. No la expongas a Internet. La cámara del navegador requiere permiso y funciona en `localhost`; si falla, usa carga de archivo.

## 7. Ejecutar validaciones automatizadas

Compilación estática de módulos:

```powershell
python -m compileall -q src tests
```

Suite completa:

```powershell
python -m unittest discover -s tests -v
```

Estas pruebas crean únicamente datos temporales o bases efímeras controladas por los tests.

## 8. Smoke test de la demo

Con la web ejecutándose en otra terminal:

```powershell
python scripts/run_web_demo_smoke_test.py --health-only
```

Para el flujo HTTP completo, usa imágenes locales con consentimiento y un UID nuevo:

```powershell
python scripts/run_web_demo_smoke_test.py `
  --base-url http://127.0.0.1:8000 `
  --uid 01020304 `
  --name "Usuario Demo 1" `
  --reference "C:\ruta\user_001\frame_000001.jpg" `
  --positive-capture "C:\ruta\user_001\frame_000002.jpg" `
  --negative-capture "C:\ruta\user_002\frame_000001.jpg"
```

El flujo crea un usuario, referencias y eventos en la base local. Usa una base limpia o un UID distinto. No agregues esos archivos al commit. Para cámara e iluminación difícil sigue [integrated_demo_test_plan.md](integrated_demo_test_plan.md).

## 9. Notebook de presentación

```powershell
python -m pip install jupyter
jupyter notebook notebooks/99_project_summary.ipynb
```

El notebook no entrena ni requiere SQLite. Carga JSON/CSV pequeños si están disponibles y usa valores documentados como respaldo. Las rutas están concentradas en una celda editable. Antes de guardar para entrega, reinicia el kernel y limpia todas las salidas.

## 10. Comandos reproducibles del dataset y evaluación

Estos comandos se documentan para trazabilidad; requieren datos privados locales. No los ejecutes durante la presentación:

```powershell
python -m src.dataset.build_manifest
python -m src.dataset.build_splits --seed 42
python -m src.dataset.build_pairs --seed 42 --overwrite
python -m src.dataset.audit_splits
python -m src.dataset.validate_support_set
```

Evaluación de un modelo ya entrenado:

```powershell
python -m src.evaluation.evaluate `
  --mode test `
  --experiment-name baseline_formal/baseline_con_aumento `
  --batch-size 64 `
  --device cpu
```

No recalibres el threshold final para una demo. La calibración ya documentada usó únicamente validation.

## 11. Regeneración excepcional en WSL/GPU

Solo es necesaria si se pierde el artefacto privado o se define un experimento nuevo. Se realiza manualmente en WSL2 con GPU configurada y una copia privada del dataset. La preparación de WSL/driver es responsabilidad del equipo; este flujo no instala componentes del sistema ni solicita contraseñas.

En el entorno Python ya preparado dentro de WSL:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.utils.check_gpu
python scripts/train_final_model.py --device gpu
```

El último comando, sin `--execute`, solo imprime el plan. Revisa la auditoría, las rutas y los artefactos antes de autorizar una regeneración. La ejecución deliberada sería:

```bash
python scripts/train_final_model.py --device gpu --execute
```

Si ya existen artefactos, el script se detiene. `--overwrite` es una confirmación destructiva y solo debe usarse después de respaldar y revisar manualmente. Después de cualquier entrenamiento nuevo se deben recalibrar validation y evaluar test una sola vez; no se debe sustituir silenciosamente `config/model_config.json`.

Consulta [gpu_training_setup.md](gpu_training_setup.md) para diagnóstico de GPU. El entorno del proyecto final y la demo sigue siendo Windows nativo.

## 12. Solución de problemas

| Síntoma | Acción |
|---|---|
| `check-model` no encuentra `.keras` | Verificar `model_path`; restaurar el artefacto privado sin versionarlo. |
| Error de forma de entrada | Confirmar modelo final y `input_size=[112,112,3]`. |
| Web devuelve `INFERENCE_ERROR` | Ejecutar `check-model` y revisar consola; la decisión seguirá siendo `DENIED`. |
| Cámara bloqueada | Conceder permiso en `localhost` o usar carga de archivo. |
| Score bajo con usuario correcto | Mejorar luz/encuadre y recapturar; no bajar el threshold. |
| UID desconocido o inactivo | Revisar registro y estado del usuario. |
| UID duplicado | Usar otro UID simulado o limpiar de forma consciente la base local. |

## 13. Limpieza segura antes de commit

```powershell
git status --short
git diff --check
```

Confirma que no aparezcan `.keras`, `.sqlite*`, imágenes, videos, datasets, checkpoints, cachés o rutas personales. No uses una eliminación masiva; revisa cada archivo no rastreado antes de decidir.
