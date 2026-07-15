# Resultados del reentrenamiento baseline formal

## Alcance y validez

Las métricas históricas del proyecto no se consideran resultados finales porque se obtuvieron con el protocolo anterior, que permitía fuga entre splits. Este experimento usa el flujo corregido: manifiesto, split por video, pares generados dentro de cada split y auditoría previa al entrenamiento.

La auditoría confirmó cero imágenes compartidas, cero videos compartidos, cero pares repetidos y cero hashes cruzados. Se usaron `data/pairs/train_pairs.csv` (4000 pares), `data/pairs/val_pairs.csv` (500) y `data/pairs/test_pairs.csv` (500), todos balanceados. Validation y test permanecieron sin aumentos aleatorios.

## Configuración

- Variante entrenada: `baseline_con_aumento`.
- Arquitectura: baseline siamés existente, sin cambios estructurales.
- Train: aumentos realistas activados.
- Validation: aumentos desactivados.
- Optimizador: Adam, learning rate `0.0001`.
- Batch size: 64.
- Épocas: 6 máximas, 4 completas.
- Early stopping: `val_loss`, paciencia 2, restauración del mejor estado.
- Checkpoint: mejor `val_loss`; época 4, `0.3234343231`.
- Semilla: 42 para Python, NumPy y TensorFlow.
- Runtime: TensorFlow 2.21 en CPU sobre Windows nativo.
- Modelo local: `models/saved_model/baseline_formal/baseline_con_aumento.keras` (~50.6 MB, ignorado por Git).

La época 5 se detuvo por presión de recursos del entorno CPU tras varias horas sin avanzar. El checkpoint válido de la época 4 quedó intacto y se usó en todas las evaluaciones. Debido al costo observado, la variante `baseline_sin_aumento` queda pendiente; no se comparan variantes incompletas.

## Calibración del threshold

El threshold se seleccionó exclusivamente con validation. El criterio fue maximizar F1; ante empates, priorizar menor FAR y luego menor FRR, una decisión conservadora para control de acceso. Test y stress tests no participaron en la selección.

Threshold seleccionado: `0.0455392189`.

| Métrica validation | Resultado |
|---|---:|
| Pares (positivos / negativos) | 500 (250 / 250) |
| Accuracy | 0.9720 |
| Precision | 0.9538 |
| Recall | 0.9920 |
| F1 | 0.9725 |
| FAR | 0.0480 |
| FRR | 0.0080 |
| ROC AUC | 0.9833 |
| Matriz TN / FP / FN / TP | 238 / 12 / 2 / 248 |

## Test limpio

El test se evaluó una sola vez con el threshold ya congelado. Estas métricas reemplazan a las métricas antiguas porque corresponden al protocolo sin fuga.

| Métrica test limpio | Resultado |
|---|---:|
| Pares (positivos / negativos) | 500 (250 / 250) |
| Accuracy | 0.9760 |
| Precision | 0.9577 |
| Recall | 0.9960 |
| F1 | 0.9765 |
| FAR | 0.0440 |
| FRR | 0.0040 |
| ROC AUC | 0.9912 |
| Matriz TN / FP / FN / TP | 239 / 11 / 1 / 249 |

## Pruebas de estrés separadas

Las condiciones son deterministas (semilla 2026), usan los 500 pares de test y alteran solo `image_b`, interpretada como probe; `image_a` permanece limpia como referencia. No se mezclan con el test limpio ni modifican el threshold.

| Condición | Accuracy | F1 | FAR | FRR | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Poca luz | 0.7680 | 0.7875 | 0.3240 | 0.1400 | 0.8568 |
| Sobreexposición | 0.8800 | 0.8855 | 0.1680 | 0.0720 | 0.9552 |
| Bajo contraste | 0.9080 | 0.9125 | 0.1440 | 0.0400 | 0.9690 |
| Ruido | 0.9640 | 0.9648 | 0.0600 | 0.0120 | 0.9932 |
| Blur | 0.9620 | 0.9634 | 0.0760 | 0.0000 | 0.9884 |
| Rotación/recorte | 0.9700 | 0.9708 | 0.0560 | 0.0040 | 0.9924 |
| Oclusión parcial | 0.9740 | 0.9745 | 0.0440 | 0.0080 | 0.9931 |
| Lentes sintéticos | 0.9560 | 0.9575 | 0.0800 | 0.0080 | 0.9909 |
| Sombra de barba sintética | 0.9700 | 0.9706 | 0.0520 | 0.0080 | 0.9900 |

## Limitaciones y siguiente sesión

El dataset contiene solo 11 personas y las pruebas de estrés son sintéticas; las métricas no sustituyen una validación con usuarios y condiciones reales. El threshold es bajo, aunque coherente con la distribución de scores del checkpoint, por lo que debe volver a verificarse cuando cambie el dataset o se entrene otra variante.

La siguiente sesión recomendada es completar `baseline_sin_aumento` en WSL2/GPU o un entorno con más memoria, repetir exactamente la calibración/evaluación y comparar ambas variantes. Después conviene recolectar ejemplos reales de poca luz, sobreexposición y bajo contraste, que son las condiciones con mayor degradación.
