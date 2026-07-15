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
- Épocas: 10 máximas y 10 completas.
- Early stopping: `val_loss`, paciencia final 5.
- Checkpoint: mejor `val_loss`; época 8, `0.2569648027`.
- Semilla: 42 para Python, NumPy y TensorFlow.
- Runtime: TensorFlow 2.21 en CPU sobre Windows nativo.
- Modelo local: `models/saved_model/baseline_formal/baseline_con_aumento.keras` (~50.6 MB, ignorado por Git).

El entrenamiento se reanudó desde checkpoints completos después de suspensiones de la laptop. Se preservaron pesos, estado de Adam, mejor `val_loss` global e historial acumulado. `history.csv` contiene diez épocas completas. Las épocas 9 y 10 no mejoraron la época 8, por lo que todas las evaluaciones finales usan el checkpoint de época 8.

La variante `baseline_sin_aumento` se entrenó posteriormente desde cero con el mismo protocolo. Completó 10 épocas y su mejor checkpoint fue la época 5 (`val_loss` 0.322139). La comparación completa se encuentra en [baseline_comparison_results.md](baseline_comparison_results.md).

## Calibración del threshold

El threshold se seleccionó exclusivamente con validation. El criterio fue maximizar F1; ante empates, priorizar menor FAR y luego menor FRR, una decisión conservadora para control de acceso. Test y stress tests no participaron en la selección.

Threshold seleccionado: `0.0187880173`.

| Métrica validation | Resultado |
|---|---:|
| Pares (positivos / negativos) | 500 (250 / 250) |
| Accuracy | 0.9840 |
| Precision | 0.9764 |
| Recall | 0.9920 |
| F1 | 0.9841 |
| FAR | 0.0240 |
| FRR | 0.0080 |
| ROC AUC | 0.9853 |
| Matriz TN / FP / FN / TP | 244 / 6 / 2 / 248 |

## Test limpio

El test se evaluó con el threshold ya congelado. Estas métricas reemplazan a todas las métricas anteriores porque corresponden al checkpoint final seleccionado mediante validation y al protocolo sin fuga.

| Métrica test limpio | Resultado |
|---|---:|
| Pares (positivos / negativos) | 500 (250 / 250) |
| Accuracy | 0.9800 |
| Precision | 0.9651 |
| Recall | 0.9960 |
| F1 | 0.9803 |
| FAR | 0.0360 |
| FRR | 0.0040 |
| ROC AUC | 0.9838 |
| Matriz TN / FP / FN / TP | 241 / 9 / 1 / 249 |

## Pruebas de estrés separadas

Las condiciones son deterministas (semilla 2026), usan los 500 pares de test y alteran solo `image_b`, interpretada como probe; `image_a` permanece limpia como referencia. No se mezclan con el test limpio ni modifican el threshold.

| Condición | Accuracy | F1 | FAR | FRR | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Poca luz | 0.8440 | 0.8539 | 0.2240 | 0.0880 | 0.9204 |
| Sobreexposición | 0.9360 | 0.9380 | 0.0960 | 0.0320 | 0.9730 |
| Bajo contraste | 0.9120 | 0.9141 | 0.1120 | 0.0640 | 0.9566 |
| Ruido | 0.9640 | 0.9646 | 0.0520 | 0.0200 | 0.9863 |
| Blur | 0.9820 | 0.9822 | 0.0320 | 0.0040 | 0.9850 |
| Rotación/recorte | 0.9780 | 0.9782 | 0.0320 | 0.0120 | 0.9902 |
| Oclusión parcial | 0.9740 | 0.9745 | 0.0440 | 0.0080 | 0.9896 |
| Lentes sintéticos | 0.9800 | 0.9803 | 0.0360 | 0.0040 | 0.9854 |
| Sombra de barba sintética | 0.9820 | 0.9822 | 0.0320 | 0.0040 | 0.9872 |

## Limitaciones y siguiente sesión

El dataset contiene solo 11 personas y las pruebas de estrés son sintéticas; las métricas no sustituyen una validación con usuarios y condiciones reales. El threshold es bajo, aunque coherente con la distribución de scores del checkpoint, por lo que debe volver a verificarse cuando cambie el dataset o se entrene otra variante.

La comparación ya confirmó que los aumentos preservan el test limpio y mejoran ampliamente los stress tests. La siguiente sesión recomendada es recolectar ejemplos reales de poca luz, sobreexposición y bajo contraste, poblar un support set separado y repetir el protocolo con una segunda sesión de captura.
