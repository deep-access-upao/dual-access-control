# Comparación formal: baseline con y sin aumento

## Objetivo y protocolo controlado

Se compararon `baseline_formal/baseline_con_aumento` y `baseline_formal/baseline_sin_aumento` para medir el efecto del aumento de datos en verificación facial. La arquitectura, los splits, el optimizador Adam, learning rate `0.0001`, batch size 64, seed 42, máximo de 10 épocas, checkpoint por `val_loss` y patience 5 fueron iguales. La única diferencia experimental fue activar o desactivar augmentation en train.

La auditoría previa confirmó cero imágenes, videos, pares y hashes compartidos entre splits. Se usaron 4000 pares de train, 500 de validation y 500 de test, balanceados 50/50. Validation y test permanecieron sin aumentos. Cada threshold se eligió exclusivamente en validation maximizando F1, con desempate por menor FAR y luego menor FRR. Test limpio y stress tests no participaron en la selección.

## Entrenamiento

| Variante | Augmentation train | Épocas | Mejor época | Mejor val_loss | Modelo local |
|---|---:|---:|---:|---:|---|
| Con aumento | Sí | 10 | 8 | 0.256965 | `models/saved_model/baseline_formal/baseline_con_aumento.keras` |
| Sin aumento | No | 10 | 5 | 0.322139 | `models/saved_model/baseline_formal/baseline_sin_aumento.keras` |

Los archivos `.keras` permanecen ignorados por Git.

## Validation y calibración

| Variante | Threshold | Accuracy | Precision | Recall | F1 | FAR | FRR | ROC AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Con aumento | 0.0187880173 | 0.9840 | 0.9764 | 0.9920 | 0.9841 | 0.0240 | 0.0080 | 0.9853 | 244/6/2/248 |
| Sin aumento | 0.0459242687 | 0.9840 | 0.9764 | 0.9920 | 0.9841 | 0.0240 | 0.0080 | 0.9918 | 244/6/2/248 |

Ambas variantes alcanzaron la misma matriz de confusión al calibrar, aunque sus escalas de score y sus AUC difieren. Los thresholds no son intercambiables entre checkpoints.

## Test limpio

| Variante | Accuracy | Precision | Recall | F1 | FAR | FRR | ROC AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Con aumento | 0.9800 | 0.9651 | 0.9960 | 0.9803 | 0.0360 | 0.0040 | 0.9838 | 241/9/1/249 |
| Sin aumento | 0.9780 | 0.9761 | 0.9800 | 0.9780 | 0.0240 | 0.0200 | 0.9957 | 244/6/5/245 |

El aumento produjo una mejora pequeña de accuracy (+0.0020) y F1 (+0.0023). También redujo FRR de 0.0200 a 0.0040, pero elevó FAR de 0.0240 a 0.0360 y redujo ROC AUC. En test limpio no existe una victoria absoluta: la variante con aumento acepta mejor a usuarios legítimos, mientras la variante sin aumento rechaza mejor impostores al threshold calibrado.

## Stress tests deterministas

Las transformaciones usan seed 2026, afectan solo `image_b` (probe), mantienen `image_a` limpia y conservan el threshold calibrado en validation.

| Condición | Accuracy con | Accuracy sin | F1 con | F1 sin | FAR con | FAR sin | FRR con | FRR sin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Poca luz | 0.8440 | 0.5180 | 0.8539 | 0.2445 | 0.2240 | 0.1200 | 0.0880 | 0.8440 |
| Sobreexposición | 0.9360 | 0.4420 | 0.9380 | 0.0000 | 0.0960 | 0.1160 | 0.0320 | 1.0000 |
| Bajo contraste | 0.9120 | 0.7660 | 0.9141 | 0.7181 | 0.1120 | 0.0640 | 0.0640 | 0.4040 |
| Ruido | 0.9640 | 0.9580 | 0.9646 | 0.9572 | 0.0520 | 0.0240 | 0.0200 | 0.0600 |
| Blur | 0.9820 | 0.8700 | 0.9822 | 0.8571 | 0.0320 | 0.0400 | 0.0040 | 0.2200 |
| Rotación/recorte | 0.9780 | 0.9760 | 0.9782 | 0.9761 | 0.0320 | 0.0280 | 0.0120 | 0.0200 |
| Oclusión parcial | 0.9740 | 0.9740 | 0.9745 | 0.9741 | 0.0440 | 0.0280 | 0.0080 | 0.0240 |
| Lentes sintéticos | 0.9800 | 0.9780 | 0.9803 | 0.9780 | 0.0360 | 0.0240 | 0.0040 | 0.0200 |
| Sombra de barba sintética | 0.9820 | 0.9720 | 0.9822 | 0.9719 | 0.0320 | 0.0240 | 0.0040 | 0.0320 |
| **Promedio** | **0.9502** | **0.8282** | **0.9520** | **0.7419** | **0.0733** | **0.0520** | **0.0262** | **0.2916** |

El peor caso con aumento fue poca luz (accuracy 0.8440, F1 0.8539); sin aumento fue sobreexposición (accuracy 0.4420, F1 0.0000 y FRR 1.0000). El aumento mejora de forma marcada la robustez, sobre todo ante iluminación, contraste y blur. Responde favorablemente a la observación del profesor sobre realismo: entrenar sin aumentos conserva el test limpio, pero falla al cambiar las condiciones de captura.

## Conclusión y modelo recomendado

Conviene continuar con `baseline_con_aumento`: mantiene el rendimiento limpio y es mucho más robusto en stress. La recomendación no oculta el riesgo de seguridad: su FAR limpio es 0.036 frente a 0.024 sin aumento y bajo poca luz llega a 0.224. Antes de una demo de control de acceso deben recolectarse casos reales difíciles, revisar el equilibrio FAR/FRR y volver a calibrar solo con validation representativa.

## Artefactos y reproducibilidad

- `outputs/experiments/baseline_formal/comparison/comparison_summary.json`
- `outputs/experiments/baseline_formal/comparison/clean_test_comparison.csv`
- `outputs/experiments/baseline_formal/comparison/clean_test_comparison.png`
- `outputs/experiments/baseline_formal/comparison/stress_comparison.png`
- `outputs/experiments/baseline_formal/baseline_sin_aumento/` contiene historial, configuración, métricas, predicciones y gráficos livianos.

La comparación se regenera con `python -m src.evaluation.compare_baselines` después de evaluar ambas variantes.

## Limitaciones y próxima sesión

- Solo se evaluaron 11 personas y el dataset es pequeño.
- El support set está vacío; todavía no se validó el flujo one-shot/few-shot de usuarios autorizados.
- Los stress tests son sintéticos y no sustituyen una segunda sesión real.
- Poca luz sigue siendo la debilidad principal de la variante recomendada, especialmente por FAR.
- Se necesita una segunda captura real con iluminación, exposición, distancia, pose y oclusión variadas; luego deben repetirse splits por video, auditoría, entrenamiento y calibración sin tocar test.
