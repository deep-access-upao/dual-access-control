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
- Épocas: 6 máximas y 6 completas.
- Early stopping: `val_loss`, paciencia 2.
- Checkpoint: mejor `val_loss`; época 6, `0.2735352516`.
- Semilla: 42 para Python, NumPy y TensorFlow.
- Runtime: TensorFlow 2.21 en CPU sobre Windows nativo.
- Modelo local: `models/saved_model/baseline_formal/baseline_con_aumento.keras` (~50.6 MB, ignorado por Git).

Después de una suspensión de la laptop, las épocas 5 y 6 se reanudaron desde el checkpoint completo de la época 4. Antes de continuar se verificaron las 252 iteraciones de Adam (`63 pasos × 4 épocas`), se anexó el nuevo historial al existente y se preservó el mejor `val_loss` global. El entrenamiento terminó con código 0 y seis filas válidas en `history.csv`.

Debido al costo CPU observado, la variante `baseline_sin_aumento` queda pendiente; no se comparan variantes incompletas.

## Calibración del threshold

El threshold se seleccionó exclusivamente con validation. El criterio fue maximizar F1; ante empates, priorizar menor FAR y luego menor FRR, una decisión conservadora para control de acceso. Test y stress tests no participaron en la selección.

Threshold seleccionado: `0.0157270245`.

| Métrica validation | Resultado |
|---|---:|
| Pares (positivos / negativos) | 500 (250 / 250) |
| Accuracy | 0.9820 |
| Precision | 0.9725 |
| Recall | 0.9920 |
| F1 | 0.9822 |
| FAR | 0.0280 |
| FRR | 0.0080 |
| ROC AUC | 0.9880 |
| Matriz TN / FP / FN / TP | 243 / 7 / 2 / 248 |

## Test limpio

El test se evaluó con el threshold ya congelado. Estas métricas reemplazan a las métricas antiguas porque corresponden al protocolo sin fuga.

| Métrica test limpio | Resultado |
|---|---:|
| Pares (positivos / negativos) | 500 (250 / 250) |
| Accuracy | 0.9740 |
| Precision | 0.9575 |
| Recall | 0.9920 |
| F1 | 0.9745 |
| FAR | 0.0440 |
| FRR | 0.0080 |
| ROC AUC | 0.9883 |
| Matriz TN / FP / FN / TP | 239 / 11 / 2 / 248 |

## Pruebas de estrés separadas

Las condiciones son deterministas (semilla 2026), usan los 500 pares de test y alteran solo `image_b`, interpretada como probe; `image_a` permanece limpia como referencia. No se mezclan con el test limpio ni modifican el threshold.

| Condición | Accuracy | F1 | FAR | FRR | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Poca luz | 0.8060 | 0.8194 | 0.2680 | 0.1200 | 0.9124 |
| Sobreexposición | 0.9120 | 0.9163 | 0.1400 | 0.0360 | 0.9710 |
| Bajo contraste | 0.8980 | 0.9002 | 0.1240 | 0.0800 | 0.9583 |
| Ruido | 0.9520 | 0.9533 | 0.0760 | 0.0200 | 0.9887 |
| Blur | 0.9780 | 0.9785 | 0.0440 | 0.0000 | 0.9856 |
| Rotación/recorte | 0.9720 | 0.9724 | 0.0440 | 0.0120 | 0.9922 |
| Oclusión parcial | 0.9700 | 0.9706 | 0.0520 | 0.0080 | 0.9920 |
| Lentes sintéticos | 0.9740 | 0.9745 | 0.0440 | 0.0080 | 0.9893 |
| Sombra de barba sintética | 0.9760 | 0.9764 | 0.0400 | 0.0080 | 0.9882 |

## Limitaciones y siguiente sesión

El dataset contiene solo 11 personas y las pruebas de estrés son sintéticas; las métricas no sustituyen una validación con usuarios y condiciones reales. El threshold es bajo, aunque coherente con la distribución de scores del checkpoint, por lo que debe volver a verificarse cuando cambie el dataset o se entrene otra variante.

La siguiente sesión recomendada es entrenar `baseline_sin_aumento` en WSL2/GPU o un entorno con más memoria, repetir exactamente la calibración/evaluación y comparar ambas variantes. Después conviene recolectar ejemplos reales de poca luz, sobreexposición y bajo contraste, que son las condiciones con mayor degradación.
