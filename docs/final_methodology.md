# Metodología final del proyecto

## 1. Formulación del problema

El proyecto aborda la verificación de identidad en un control de acceso de doble factor. Un UID RFID identifica un registro, pero no prueba que la persona que porta la credencial sea su titular. El segundo factor compara una captura facial con las referencias asociadas a ese usuario y concede acceso únicamente cuando ambos factores son válidos.

La verificación facial presenta variaciones no lineales de iluminación, pose, escala, encuadre, calidad de cámara, oclusión y apariencia. Diseñar reglas manuales para todas ellas no es viable. Una red siamesa aprende representaciones compartidas y una función de similitud entre dos imágenes. A diferencia de un clasificador cerrado, permite registrar referencias nuevas sin reentrenar una clase por usuario, lo que encaja con el support set de la demo.

El impacto esperado es demostrar un flujo biométrico reproducible y auditable para fines académicos. Su viabilidad se acota a una demo local: no se afirma preparación productiva ni se sustituye una evaluación biométrica a gran escala.

## 2. Objetivo e indicadores

El objetivo general fue construir un prototipo reproducible que asocie un UID con un usuario, verifique su rostro con una red siamesa y registre una decisión de acceso, manteniendo FAR de test limpio en 2% con el threshold de seguridad seleccionado.

Los indicadores cuantitativos fueron accuracy, precision, recall, F1, FAR, FRR, ROC AUC y matriz de confusión. El indicador funcional fue completar en la web los flujos de registro, `GRANTED`, rechazos por UID desconocido, usuario inactivo y rostro no coincidente, además del historial.

## 3. Datos y preprocesamiento

### 3.1 Composición

- 11 personas y 44 videos, cuatro videos por persona.
- 1,702 imágenes detectadas originalmente.
- 1,544 imágenes utilizables y 158 rechazadas por controles de calidad.
- 739 imágenes/22 videos en train, 400 imágenes/11 videos en validation y 405 imágenes/11 videos en test.
- 4,000 pares de train, 500 de validation y 500 de test, balanceados 50/50.

Los datos privados siguen conceptualmente esta estructura:

```text
data/
  raw/<persona>/<vista>/<video>
  processed/<persona>/<vista>/<frames>
  pairs/{train_pairs,val_pairs,test_pairs}.csv
  support_set/<usuario>/<referencias>.jpg
```

Estas carpetas no se versionan. El repositorio conserva únicamente marcadores y código reproducible.

### 3.2 Transformaciones

El pipeline detecta y recorta el rostro, descarta muestras que no cumplen controles de calidad y registra metadatos y SHA-256 en un manifest local. En carga se decodifica JPEG, se usa RGB, se redimensiona a `112x112x3`, se convierte a `float32` y se divide entre 255 para obtener valores en `[0, 1]`.

Los pares positivos contienen dos imágenes de la misma persona; los negativos, de personas diferentes. Cada rama recibe su imagen y la etiqueta binaria representa coincidencia/no coincidencia.

### 3.3 Aumento de datos

Train incorpora rotación moderada, traslación, zoom, brillo, contraste, gamma, ruido, blur, degradación de resolución y oclusión pequeña. Se aplican en memoria y con semillas controladas. Validation y test siempre se mantienen limpios. Los stress tests son evaluaciones deterministas separadas y no forman parte del entrenamiento.

## 4. División y prevención de fuga

El enfoque inicial de crear todos los pares y dividirlos aleatoriamente era incorrecto: una imagen o frames casi idénticos de un mismo video podían aparecer en varios splits. La red podía memorizar condiciones de captura en vez de generalizar, y las métricas quedaban infladas.

El flujo corregido es:

```text
manifest -> split por video -> pares dentro de cada split -> auditoría
```

El video es la unidad indivisible. Para cada persona se asignaron dos videos a train, uno a validation y uno a test. Las identidades aparecen en los tres splits, pero ningún video ni frame cruza particiones. Esta decisión mide generalización entre sesiones/vistas de las mismas identidades; no equivale todavía a evaluar identidades completamente no vistas.

La auditoría comprobó cero imágenes compartidas, cero videos compartidos, cero pares repetidos y cero hashes cruzados. También valida columnas, etiquetas y pertenencia de cada imagen al split declarado.

## 5. Diseño experimental

El baseline formal usa una arquitectura siamesa propia. La comparación con/sin aumento mantuvo constantes arquitectura, datos, optimizer, learning rate, batch, seed, máximo de épocas, checkpoint y paciencia. La única variable fue el aumento de train.

Configuración final:

| Componente | Valor |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 64 |
| Épocas máximas/completadas | 10/10 |
| Early stopping | `val_loss`, patience 5 |
| Mejor checkpoint | época 8, `val_loss=0.2569648` |
| Seed de entrenamiento | 42 |
| Seed de stress | 2026 |
| Augmentation | solo train |
| Validation/test | limpios |

Cada experimento conserva configuración, historial, predicciones, métricas y gráficos bajo `outputs/experiments/`. El modelo privado queda bajo `models/saved_model/` y no se versiona. Test no se usa durante entrenamiento ni para elegir threshold.

## 6. Arquitectura y técnicas de Deep Learning

### 6.1 Baseline seleccionado

Las dos imágenes pasan por el mismo encoder, por lo que comparten pesos. El encoder contiene cuatro bloques `Conv2D + BatchNormalization + MaxPooling`, seguidos por `Flatten`, Dense 256, Dropout 0.3 y un embedding de 128 componentes. La distancia absoluta L1 entre embeddings alimenta una salida Dense sigmoide. El modelo tiene 4,208,257 parámetros y se optimiza con Binary Crossentropy.

No se utilizó un backbone preentrenado en el modelo final. Se priorizó estabilizar el protocolo sin fuga, realizar comparaciones controladas y completar una demo reproducible dentro del tiempo y tamaño de dataset disponibles. Esta decisión reduce dependencias experimentales, pero limita la transferencia desde conjuntos faciales más amplios.

### 6.2 Variante GAP + L2 + coseno

Se evaluó una alternativa que reemplaza `Flatten` por `GlobalAveragePooling2D`, normaliza el embedding con L2 y compara mediante coseno. Tiene 1,062,400 parámetros, 74.8% menos que el baseline. En test limpio obtuvo accuracy 0.9740, F1 0.9735, FAR 0.0080 y FRR 0.0440, ligeramente mejor que el baseline final.

No fue seleccionada porque en poca luz, sobreexposición y bajo contraste su FRR llegó a 1.000, 0.900 y 0.820. Su menor FAR no compensa el rechazo casi total de usuarios genuinos bajo las condiciones que motivaron la mejora.

### 6.3 Fine-tuning y distillation

No se aplicó fine-tuning profundo: el proyecto usa una CNN siamesa propia y un dataset pequeño. Una extensión razonable es usar MobileNetV2 preentrenado como encoder congelado, evaluar con el mismo protocolo y después liberar selectivamente los últimos bloques con learning rate reducido.

Knowledge distillation no se implementó. No existía una pareja teacher-student validada ni una restricción de despliegue que justificara priorizarla. Corregir fuga, medir robustez, calibrar seguridad y completar la demo aportaban más valor metodológico. Queda como opción futura si se entrena un teacher superior y se requiere un modelo más pequeño.

## 7. Calibración del threshold

La regla es `score >= threshold`. Los thresholds se exploraron exclusivamente con validation. `max_f1=0.0187880173` maximiza F1, pero en test produce FAR 0.0360. El criterio `security_first` selecciona el menor FAR entre candidatos que cumplen simultáneamente FAR <= 2% y FRR <= 5% en validation, con desempate por F1.

El valor final es `0.3128704727`. En validation obtuvo FAR 0.0200, FRR 0.0280 y F1 0.9759. Test solo se consultó después: FAR 0.0200, FRR 0.0480 y F1 0.9655. El cambio reduce falsos positivos de test de 9 a 5, a costa de aumentar falsos negativos de 1 a 12. Para un control de acceso, se documentó esa prioridad de seguridad sin ocultar la fricción adicional.

## 8. Resultados y selección final

### 8.1 Test limpio final

| Accuracy | Precision | Recall | F1 | FAR | FRR | ROC AUC | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9660 | 0.9794 | 0.9520 | 0.9655 | 0.0200 | 0.0480 | 0.9838 | 245 | 5 | 12 | 238 |

### 8.2 Con aumento vs sin aumento

Con sus thresholds `max_f1`, el baseline aumentado obtuvo accuracy 0.9800 y F1 0.9803; sin aumento, 0.9780 y 0.9780. La diferencia limpia es pequeña y el aumentado presenta mayor FAR en ese operating point. Sin embargo, el promedio de accuracy de stress fue 0.9502 con aumento frente a 0.8282 sin aumento; el FRR promedio fue 0.0262 frente a 0.2916. La robustez justificó conservar el aumento.

### 8.3 Matriz, ROC y PR

Los artefactos versionados incluyen:

- `outputs/experiments/baseline_formal/baseline_con_aumento/test_confusion_matrix.png`;
- `outputs/experiments/baseline_formal/baseline_con_aumento/test_clean_roc_curve.png`;
- `outputs/experiments/baseline_formal/baseline_con_aumento/test_clean_precision_recall_curve.png`;
- `outputs/experiments/model_comparison/` para comparación de variantes.

La selección final conserva `baseline_formal/baseline_con_aumento` con threshold `0.3128704727`.

## 9. Análisis de errores

| Condición | Accuracy | F1 | FAR | FRR |
|---|---:|---:|---:|---:|
| Poca luz | 0.782 | 0.7471 | 0.080 | 0.356 |
| Sobreexposición | 0.832 | 0.8019 | 0.016 | 0.320 |
| Bajo contraste | 0.816 | 0.7909 | 0.064 | 0.304 |
| Ruido | 0.918 | 0.9118 | 0.012 | 0.152 |
| Blur | 0.952 | 0.9508 | 0.024 | 0.072 |
| Rotación/recorte | 0.952 | 0.9506 | 0.020 | 0.076 |
| Oclusión parcial | 0.938 | 0.9353 | 0.020 | 0.104 |
| Lentes sintéticos | 0.958 | 0.9571 | 0.020 | 0.064 |
| Sombra de barba | 0.906 | 0.8985 | 0.020 | 0.168 |

Las degradaciones fotométricas elevan principalmente el FRR: usuarios legítimos pueden ser rechazados aun con RFID correcto. Operacionalmente se recomienda informar mala calidad, permitir recaptura, usar iluminación frontal y registrar varias referencias autorizadas. Como investigación futura deben evaluarse un `validation-stress` separado, CLAHE u otras normalizaciones, más sesiones reales y un encoder MobileNetV2.

## 10. Inferencia modular y demo web

`src/inference` centraliza la carga del config, validación del `.keras`, preprocesamiento, comparación y decisión. La CLI permite comprobar el artefacto, verificar un par o comparar una captura con múltiples referencias usando máximo o media. La web reutiliza este paquete; no duplica el threshold.

La demo FastAPI permite registrar usuarios y referencias, activar/desactivar usuarios, ingresar un UID, cargar o capturar una imagen y consultar hasta 200 eventos recientes. SQLite y las referencias se almacenan localmente bajo `data/demo/` y se ignoran en Git.

El lector RFID y ESP32 se omitieron por tiempo. La web simula la fuente del UID, pero conserva el contrato lógico que usaría el hardware real: `UID -> usuario -> rostro -> decisión -> historial`. Sustituir el campo de texto por lectura serial no requiere cambiar la regla de acceso.

## 11. Mejoras implementadas

1. Corrección del data leakage y auditoría por videos, imágenes, pares y hashes.
2. Aumento realista solo en train y stress tests deterministas.
3. Reentrenamiento formal del baseline y comparación con/sin aumento.
4. Calibración de threshold con criterio de seguridad.
5. Evaluación controlada de GAP + L2 + coseno y decisión de descarte.
6. Paquete modular de inferencia con configuración versionada.
7. Demo web con RFID simulado, support set e historial.
8. Pruebas unitarias, rutas web, smoke test y plan integral de evidencias.

## 12. Limitaciones y trabajo futuro

- Solo 11 personas y 44 videos; estimaciones de FAR/FRR con incertidumbre alta.
- Sin usuarios externos ni evaluación estable de identidades no vistas.
- Stress sintético; falta una segunda sesión real y un `validation-stress` independiente.
- Sensibilidad a iluminación, exposición y contraste.
- Sin backbone preentrenado final, fine-tuning, distillation ni detección de vida.
- RFID simulado y sin respuesta física ESP32.
- Demo local sin autenticación, HTTPS, roles, despliegue público o hardening.

La siguiente iteración debería recolectar más personas y sesiones, separar validation-stress, evaluar calidad de captura/CLAHE, comparar MobileNetV2 congelado y con fine-tuning selectivo, añadir liveness y conectar RFID físico. Toda calibración nueva debe usar validation y reservar test para una sola evaluación final.

## 13. Reproducibilidad

La seed principal es 42 y la de stress 2026. `config/model_config.json` fija nombre, ruta, entrada, normalización, threshold y regla. `requirements.txt` declara dependencias. Los comandos completos están en [final_usage_guide.md](final_usage_guide.md). La regeneración deliberada se orquesta con `scripts/train_final_model.py`, que por defecto solo muestra el plan y exige confirmaciones para ejecutar o sobrescribir.

## 14. Ética y uso responsable

Las imágenes faciales son datos sensibles. Deben existir consentimiento, propósito limitado, control de acceso, retención mínima y eliminación segura. No se versionan imágenes, videos, support sets, bases, modelos ni capturas. El dataset pequeño puede introducir sesgos demográficos y de condiciones de adquisición. FAR y FRR representan daños distintos: acceso indebido y rechazo injustificado. El sistema es académico; cualquier uso real exige validación externa, análisis de equidad, seguridad contra presentación, cumplimiento legal y mecanismos de revisión humana.
