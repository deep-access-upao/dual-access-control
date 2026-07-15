# Mejora controlada del modelo siamés

> La comparación se conserva aquí con todo su detalle. La decisión de selección y las técnicas no implementadas se integran en la [metodología final](final_methodology.md) y la [matriz de rúbrica](rubric_coverage.md).

## Objetivo y protocolo

La sesión 5 evaluó si una arquitectura con `GlobalAveragePooling2D`, embedding de 128 dimensiones normalizado con L2 y similitud coseno mejora el baseline formal. El problema objetivo no era reducir un FAR limpio que ya era razonable, sino bajar el FRR bajo poca luz, sobreexposición y bajo contraste sin degradar la seguridad.

Se trabajó exclusivamente en WSL/Ubuntu con el entorno Conda `dual-access-gpu` y la RTX 2060. No se modificaron dataset, manifest, splits ni pares. Train conservó augmentation; validation y test se cargaron limpios. Los thresholds se eligieron solo con validation. Test limpio y test-stress se ejecutaron después de fijar el candidato y no participaron en la calibración.

El proyecto no dispone de `validation-stress`. Por ello, test-stress se reporta como evidencia secundaria y no puede demostrar por sí solo una mejora de selección. Ante esa limitación y la falta de mejora fotométrica, el cambio de modelo se rechaza de forma conservadora.

## Diagnóstico y arquitecturas

El baseline `baseline_formal/baseline_con_aumento` usa cuatro bloques Conv2D + BatchNorm + MaxPool, `Flatten`, Dense 256, Dropout 0.3, embedding 128 sin normalizar, distancia L1 y una cabeza Dense sigmoide. Tiene 4.208.257 parámetros.

La variante `baseline_formal/siamese_gap_l2_cosine` conserva los cuatro bloques CNN y reemplaza `Flatten` por GAP. Después usa Dense 256, Dropout 0.3, Dense 128 y `UnitNormalization`. Los embeddings compartidos se comparan con producto punto, que equivale al coseno al tener norma unitaria, y se transforma de `[-1, 1]` a `[0, 1]` para mantener Binary Crossentropy. Tiene 1.062.400 parámetros, aproximadamente 74,8% menos que el baseline.

| Modelo | Agregación | Embedding | Comparación | Parámetros |
|---|---|---|---|---:|
| baseline_con_aumento | Flatten | 128 sin normalizar | L1 + Dense sigmoid | 4.208.257 |
| siamese_gap_l2_cosine | GAP | 128 normalizado L2 | coseno reescalado | 1.062.400 |

## Configuración de entrenamiento

- CSV: `train_pairs.csv` para entrenamiento y `val_pairs.csv` para validación.
- Seed: 42.
- Batch size: 64; no fue necesario reducirlo.
- Learning rate: 0.0001 con Adam.
- Máximo: 10 épocas; patience 5 sobre `val_loss`.
- Augmentation: activado solo en train.
- Dispositivo: `gpu` obligatorio.
- Épocas completadas: 10; mejor checkpoint en época 9, `val_loss=0.340268`.
- Modelo guardado localmente en `models/saved_model/baseline_formal/siamese_gap_l2_cosine.keras`; no se incluye en Git.

## Validation y thresholds

ROC AUC en validation: baseline 0.985328; variante 0.999056.

| Modelo | Criterio | Threshold | Acc. | Precision | Recall | F1 | FAR | FRR | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | max_f1 | 0.0187880173 | 0.9840 | 0.9764 | 0.9920 | 0.9841 | 0.0240 | 0.0080 | 244/6/2/248 |
| baseline | security_first | 0.3128704727 | 0.9760 | 0.9798 | 0.9720 | 0.9759 | 0.0200 | 0.0280 | 245/5/7/243 |
| GAP+L2+coseno | max_f1 | 0.9551179409 | 0.9900 | 1.0000 | 0.9800 | 0.9899 | 0.0000 | 0.0200 | 250/0/5/245 |
| GAP+L2+coseno | security_first | 0.9551179409 | 0.9900 | 1.0000 | 0.9800 | 0.9899 | 0.0000 | 0.0200 | 250/0/5/245 |

En la variante, `max_f1` y `security_first` convergen al mismo operating point: cumple FAR ≤ 2% y FRR ≤ 5%, y entre los thresholds factibles ofrece FAR cero con el mejor F1.

## Test limpio con security_first

| Modelo | Threshold | Acc. | Precision | Recall | F1 | FAR | FRR | ROC AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.3128704727 | 0.9660 | 0.9794 | 0.9520 | 0.9655 | 0.0200 | 0.0480 | 0.9838 | 245/5/12/238 |
| GAP+L2+coseno | 0.9551179409 | 0.9740 | 0.9917 | 0.9560 | 0.9735 | 0.0080 | 0.0440 | 0.9981 | 248/2/11/239 |

En limpio la variante mejora accuracy y F1 en 0,8 puntos porcentuales, reduce FAR en 1,2 puntos y reduce FRR en 0,4 puntos. El threshold numérico no se compara directamente con el baseline porque la escala de scores cambió.

## Stress tests con security_first

Cada condición altera solo `image_b` de los 500 pares de test, con seed 2026. Los valores son evaluación secundaria; no se usaron para calibrar ni para explorar más variantes.

| Condición | Base acc. | Base F1 | Base FAR | Base FRR | GAP acc. | GAP F1 | GAP FAR | GAP FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Poca luz | 0.782 | 0.7471 | 0.080 | 0.356 | 0.500 | 0.0000 | 0.000 | 1.000 |
| Sobreexposición | 0.832 | 0.8019 | 0.016 | 0.320 | 0.540 | 0.1786 | 0.020 | 0.900 |
| Bajo contraste | 0.816 | 0.7909 | 0.064 | 0.304 | 0.590 | 0.3051 | 0.000 | 0.820 |
| Ruido | 0.918 | 0.9118 | 0.012 | 0.152 | 0.942 | 0.9389 | 0.008 | 0.108 |
| Blur | 0.952 | 0.9508 | 0.024 | 0.072 | 0.886 | 0.8725 | 0.008 | 0.220 |
| Rotación/recorte | 0.952 | 0.9506 | 0.020 | 0.076 | 0.968 | 0.9673 | 0.012 | 0.052 |
| Oclusión parcial | 0.938 | 0.9353 | 0.020 | 0.104 | 0.960 | 0.9592 | 0.020 | 0.060 |
| Lentes sintéticos | 0.958 | 0.9571 | 0.020 | 0.064 | 0.966 | 0.9652 | 0.012 | 0.056 |
| Sombra de barba | 0.906 | 0.8985 | 0.020 | 0.168 | 0.964 | 0.9631 | 0.012 | 0.060 |

La peor condición de ambos modelos es poca luz. La variante empeora el FRR fotométrico en 64,4 puntos en poca luz, 58,0 en sobreexposición y 51,6 en bajo contraste. El bajo FAR de esos casos no compensa el rechazo casi total de genuinos. Sí mejora ruido, rotación/recorte, oclusión, lentes y sombra de barba, pero falla precisamente en el objetivo principal.

## Selección final

Se conserva `baseline_formal/baseline_con_aumento` con `security_first=0.3128704727`.

La variante cumple los controles limpios y reduce notablemente la complejidad, pero no demuestra la mejora objetivo en validation porque no existe validation-stress y, como evidencia secundaria, test-stress muestra una degradación fotométrica severa. Cambiar el modelo trasladaría el sistema hacia menos intrusiones a costa de rechazar entre 82% y 100% de los pares genuinos bajo las tres condiciones críticas.

## Variante fotométrica y MobileNetV2

No se entrenó `siamese_gap_l2_cosine_photometric`. CLAHE consistente exige definir canal de luminancia, parámetros, implementación reproducible en `tf.data` y paridad exacta con la futura inferencia web. Introducir ahora `tf.py_function`/OpenCV añadiría una dependencia CPU y riesgo de divergencia entre entrenamiento e inferencia. Queda como propuesta defendible: aplicar CLAHE suave al canal L o Y de ambas imágenes, en todos los splits y en inferencia, con parámetros fijados y pruebas de paridad antes de entrenar.

No se intentó MobileNetV2. Tras observar test-stress, abrir otra búsqueda de arquitectura usaría test de forma oportunista. Además requiere pesos ImageNet y su preprocesamiento específico. Debe evaluarse en una sesión futura pre-registrada, idealmente después de crear validation-stress real o sintético separado; entonces puede probarse congelado sin fine-tuning profundo.

## Artefactos

- Variante: `outputs/experiments/baseline_formal/siamese_gap_l2_cosine/`.
- Comparación global: `outputs/experiments/model_comparison/`.
- Notebook liviano: `notebooks/06_model_improvement_comparison.ipynb`.
- Implementación: `src/models/siamese_variants.py`.
- Comparador: `src/evaluation/compare_model_variants.py`.

Se incluyen JSON/CSV/PNG pequeños y auditables. El modelo `.keras`, datasets, imágenes, videos y checkpoints no se agregan al commit.

## Comandos reales

Todos se ejecutaron dentro de `/home/carlos/proyectos/dual-access-control` tras activar `dual-access-gpu`.

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m src.dataset.audit_splits
python -m src.dataset.validate_support_set
python -m src.utils.check_gpu

python -m src.training.train \
  --experiment-name baseline_formal/siamese_gap_l2_cosine \
  --model-variant gap_l2_cosine --augmentation --epochs 10 \
  --batch-size 64 --patience 5 --learning-rate 0.0001 \
  --seed 42 --device gpu

python -m src.evaluation.evaluate --mode calibrate \
  --experiment-name baseline_formal/siamese_gap_l2_cosine \
  --batch-size 64 --criterion max_f1 --device gpu

python -m src.evaluation.evaluate --mode test \
  --experiment-name baseline_formal/siamese_gap_l2_cosine \
  --batch-size 64 --device gpu

python -m src.evaluation.calibrate_security_thresholds \
  --experiment-name baseline_formal/siamese_gap_l2_cosine \
  --batch-size 64 --seed 2026 --device gpu --refresh-stress-predictions

python -m src.evaluation.evaluate_stress \
  --experiment-name baseline_formal/siamese_gap_l2_cosine \
  --batch-size 64 --seed 2026 --threshold 0.95511794090271 --device gpu

python -m src.evaluation.compare_model_variants
```

## Limitaciones y próxima sesión

- Solo hay 250 pares negativos y 250 positivos por split; las tasas cambian en pasos de 0,4 puntos porcentuales.
- Test-stress es sintético y no reemplaza capturas reales del lugar de despliegue.
- Falta validation-stress separado, por lo que no se puede seleccionar una arquitectura por robustez fotométrica sin consultar test.
- La salida coseno reescalada está concentrada cerca de 1 y produce un threshold alto; esto no es un problema por sí solo, pero exige usar el threshold asociado al modelo correcto.

La próxima sesión recomendada es construir un validation-stress separado y pre-registrado, incorporar capturas reales de iluminación difícil sin alterar test, y evaluar una normalización fotométrica consistente. Después puede compararse MobileNetV2 congelado bajo el mismo protocolo.
