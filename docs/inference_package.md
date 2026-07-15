# Paquete de inferencia del modelo final

## Artefacto y operating point congelados

- Modelo: `baseline_formal/baseline_con_aumento`.
- Ruta local esperada: `models/saved_model/baseline_formal/baseline_con_aumento.keras`.
- Threshold `security_first`: `0.3128704727`.
- Regla facial: `score >= threshold` produce `MATCH`; cualquier score menor produce `NO_MATCH`.
- Origen del threshold: calibración exclusiva con validation en la sesión 4C. Test y stress no se usaron para seleccionarlo.

El `.keras` es un artefacto local ignorado por Git. `config/model_config.json` contiene su identidad, ruta relativa, shape, preprocesamiento y threshold, pero no contiene el modelo ni secretos.

## Preprocesamiento

`src.inference.image_preprocessing` reutiliza `src.dataset.dataloader.load_image`, la misma función empleada durante entrenamiento y evaluación: decodificación JPEG en RGB, resize a `112 × 112`, conversión a `float32`, división entre 255 y batch con shape `(N, 112, 112, 3)`. No se usa OpenCV, por lo que no existe una conversión BGR implícita.

Riesgo conocido: el validador histórico del support set permite extensiones `.png`, pero el pipeline efectivo del modelo usa `tf.image.decode_jpeg`. El paquete conserva el comportamiento probado y devuelve un error claro ante un archivo que no pueda decodificar; ampliar formatos exige primero validar de extremo a extremo que el preprocesamiento no cambie los scores.

## Comprobación del artefacto

Desde la raíz del repositorio:

```powershell
python -m src.inference.cli check-model --config config/model_config.json
```

El comando valida JSON, threshold, ruta, carga Keras, dos entradas `(None, 112, 112, 3)` y salida `(None, 1)`. Si falta el `.keras`, termina con código distinto de cero y un mensaje explícito; nunca simula un resultado.

## Verificar dos imágenes

```powershell
python -m src.inference.cli verify-pair `
  --reference data/support_set/usuario/frontal.jpg `
  --capture ruta/captura.jpg `
  --config config/model_config.json
```

API equivalente:

```python
from src.inference import FaceVerifier

verifier = FaceVerifier.from_config("config/model_config.json")
result = verifier.verify_pair("referencia.jpg", "captura.jpg")
print(result.to_dict())
```

El resultado contiene `score`, `threshold`, `match`, `decision`, `model_name`, `reference_image` y `capture_image`.

## Verificar contra varias referencias

La estrategia recomendada es el máximo, coherente con el support set multi-vista del proyecto:

```powershell
python -m src.inference.cli verify-references `
  --capture ruta/captura.jpg `
  --references frontal.jpg left.jpg right.jpg `
  --strategy max `
  --config config/model_config.json
```

También se admite `--strategy mean`. La respuesta incluye el score de cada referencia, el score agregado y la decisión final. No se aceptan listas vacías ni scores no finitos/fuera de `[0,1]`.

## Integración futura con RFID

```python
from src.inference import decide_access

decision = decide_access(rfid_known=True, face_match=result.match)
print(decision.to_dict())
```

La regla es estricta: `RFID conocido AND rostro verificado => GRANTED`. Los motivos posibles son `RFID_UNKNOWN`, `FACE_NO_MATCH` y `RFID_AND_FACE_OK`. Esta función no abre puertos seriales ni se conecta al ESP32.

## Regeneración reproducible (no ejecutar en esta sesión)

Para inspeccionar el plan sin entrenar:

```powershell
python scripts/train_final_model.py --device gpu
```

El script muestra, pero no ejecuta, el flujo con seed 42, augmentation solo en train, 10 épocas, batch 64, learning rate `0.0001`, patience 5, baseline, calibración basada en validation, evaluación limpia y calibración `security_first`. Una ejecución real requiere `--execute`; si ya existen artefactos, exige además `--overwrite`. Debe correrse preferentemente en WSL2 con GPU y con una copia verificada de los datos/splits.

Después de reentrenar se debe revisar manualmente `security_threshold_calibration.json`. No se debe reutilizar automáticamente `0.3128704727` con un checkpoint distinto ni actualizar la configuración usando test.

Outputs esperados:

- `models/saved_model/baseline_formal/baseline_con_aumento.keras`;
- `outputs/experiments/baseline_formal/baseline_con_aumento/training_config.json` e historial;
- predicciones y métricas de validation/test;
- `security_threshold_calibration/security_threshold_calibration.json` y tablas de stress.

## Limitaciones y seguridad

- El peor caso observado continúa siendo iluminación difícil. En poca luz aumentan rechazos y todavía existe riesgo de falsas aceptaciones; se necesita control de calidad y reintento de captura.
- Las métricas proceden de 11 personas y condiciones de stress sintéticas; no equivalen a una validación de producción.
- El módulo compara rostros ya preparados; no incorpora detección de vida, anti-spoofing, detección facial ni alineación en tiempo real.
- Un `MATCH` facial aislado nunca debe conceder acceso: debe combinarse con RFID conocido.
- Cambiar modelo, dataset o preprocesamiento invalida la asociación con el threshold actual.

## Privacidad biométrica

Las imágenes faciales, support sets, videos, pares, modelos y logs con rutas o scores vinculables son datos sensibles. Deben mantenerse fuera de Git, con acceso mínimo, retención definida, consentimiento informado y cifrado apropiado. La futura web no debe exponer rutas locales ni registrar imágenes por defecto; los errores públicos deben evitar filtrar información de otros usuarios.
