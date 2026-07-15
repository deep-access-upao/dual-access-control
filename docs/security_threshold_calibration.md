# Calibración del threshold orientada a seguridad

## Contexto

El baseline seleccionado sigue siendo `baseline_formal/baseline_con_aumento` porque conserva el rendimiento en test limpio y supera ampliamente a la variante sin aumento en las pruebas de stress. Su threshold histórico, `0.0187880173`, fue elegido para maximizar F1 en validation. Ese criterio es útil como referencia general, pero no representa por sí solo el costo asimétrico de un control de acceso: un falso positivo acepta a un impostor, mientras un falso negativo obliga a un usuario legítimo a repetir la verificación o recurrir a un flujo alternativo.

Esta sesión no entrenó modelos, no alteró arquitectura, datos, splits ni pares y no sobrescribió los resultados históricos. Todos los candidatos se seleccionaron únicamente con las 500 parejas de validation. Test limpio y las nueve condiciones de stress se consultaron después, solo para evaluar los thresholds ya fijados.

## Política de selección

La regla de decisión acepta una pareja cuando `score >= threshold`. Se evaluaron estos criterios:

- `max_f1`: máximo F1 en validation, con desempate por menor FAR y luego menor FRR.
- `balanced_far_frr`: mínima diferencia absoluta entre FAR y FRR, aproximando el punto de error igual.
- `far_lte_1`, `far_lte_2` y `far_lte_3`: mejor F1 entre los thresholds cuyo FAR en validation no supera 1%, 2% o 3%, respectivamente.
- `security_first`: menor FAR entre thresholds que cumplen simultáneamente FAR ≤ 2% y FRR ≤ 5% en validation; el desempate favorece mayor F1. Si no existe uno, usa el mejor F1 con FAR ≤ 2%, luego FAR ≤ 3% y finalmente el menor FAR disponible.

La política `security_first` limita el riesgo de intrusión sin permitir que el rechazo de usuarios legítimos crezca sin control.

## Comparación de candidatos

| Criterio | Threshold | Val FAR | Val FRR | Val F1 | Test FAR | Test FRR | Test F1 | Test acc. | FP/FN test | Peor stress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| max_f1 | 0.018788 | 2.4% | 0.8% | 0.9841 | 3.6% | 0.4% | 0.9803 | 0.9800 | 9 / 1 | poca luz |
| balanced_far_frr | 0.153522 | 2.4% | 2.4% | 0.9760 | 2.0% | 1.6% | 0.9820 | 0.9820 | 5 / 4 | poca luz |
| far_lte_1 | 0.829985 | 0.8% | 81.6% | 0.3087 | 1.2% | 72.8% | 0.4237 | 0.6300 | 3 / 182 | poca luz |
| far_lte_2 | 0.312870 | 2.0% | 2.8% | 0.9759 | 2.0% | 4.8% | 0.9655 | 0.9660 | 5 / 12 | poca luz |
| far_lte_3 | 0.018788 | 2.4% | 0.8% | 0.9841 | 3.6% | 0.4% | 0.9803 | 0.9800 | 9 / 1 | poca luz |
| security_first | **0.312870** | **2.0%** | **2.8%** | **0.9759** | **2.0%** | **4.8%** | **0.9655** | **0.9660** | **5 / 12** | **poca luz** |

Subir el threshold histórico a `0.3128704727` reduce los falsos positivos de test de 9 a 5, a costa de aumentar los falsos negativos de 1 a 12. `balanced_far_frr` obtuvo el mejor resultado observado en test limpio, pero ese dato no se usó para elegirlo; en validation no cumple el límite moderado de FAR de 2%.

El criterio de 1% sí encontró un operating point, pero no es operativo: rechaza 204 de 250 parejas genuinas en validation. La separación de scores actual no permite alcanzar a la vez FAR ≤ 1% y un FRR razonable.

## Recomendación para RFID + rostro

Para la demo se recomienda `threshold = 0.3128704727` (`security_first`). Es un compromiso defendible porque cumple en validation FAR 2.0% y FRR 2.8%, y en test limpio conserva FAR 2.0%. Además, el factor RFID reduce el alcance de una aceptación facial aislada: el acceso debe concederse solo cuando ambos factores sean válidos. Esto no elimina la necesidad de controlar el FAR facial.

- Máximo F1 y referencia histórica: `0.0187880173`.
- Seguridad moderada y demo RFID + rostro: `0.3128704727`.
- Demo menos estricta: `0.1535217762`, con errores balanceados y mayor comodidad.
- Seguridad estricta experimental: `0.8299853802`; no se recomienda para operación por su FRR extremo.

No conviene escoger `max_f1` automáticamente: F1 combina precision y recall, pero no impone un techo directo al FAR. En control de acceso el falso positivo tiene un costo de seguridad mayor que el falso negativo. El FRR, por otro lado, representa fricción, reintentos, soporte manual y posible abandono por parte de usuarios legítimos.

## Métricas del threshold recomendado

### Validation (único split de calibración)

| Accuracy | Precision | Recall | F1 | FAR | FRR | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9760 | 0.9798 | 0.9720 | 0.9759 | 0.0200 | 0.0280 | 245 | 5 | 7 | 243 |

### Test limpio (solo evaluación)

| Accuracy | Precision | Recall | F1 | FAR | FRR | ROC AUC | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9660 | 0.9794 | 0.9520 | 0.9655 | 0.0200 | 0.0480 | 0.9838 | 245 | 5 | 12 | 238 |

ROC AUC no depende del threshold; se informa como medida global del ranking de scores.

## Stress tests del threshold recomendado

Estas condiciones usan seed 2026, alteran solo `image_b` (probe) y permanecen separadas del test limpio.

| Condición | Accuracy | F1 | FAR | FRR | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Poca luz | 0.7820 | 0.7471 | 0.0800 | 0.3560 | 20 | 89 |
| Sobreexposición | 0.8320 | 0.8019 | 0.0160 | 0.3200 | 4 | 80 |
| Bajo contraste | 0.8160 | 0.7909 | 0.0640 | 0.3040 | 16 | 76 |
| Ruido | 0.9180 | 0.9118 | 0.0120 | 0.1520 | 3 | 38 |
| Blur | 0.9520 | 0.9508 | 0.0240 | 0.0720 | 6 | 18 |
| Rotación/recorte | 0.9520 | 0.9506 | 0.0200 | 0.0760 | 5 | 19 |
| Oclusión parcial | 0.9380 | 0.9353 | 0.0200 | 0.1040 | 5 | 26 |
| Lentes sintéticos | 0.9580 | 0.9571 | 0.0200 | 0.0640 | 5 | 16 |
| Sombra de barba sintética | 0.9060 | 0.8985 | 0.0200 | 0.1680 | 5 | 42 |

El peor caso sigue siendo poca luz. Frente al threshold histórico, allí el FAR baja de 22.4% a 8.0%, pero el FRR sube de 8.8% a 35.6%. Esto confirma que el threshold no sustituye datos reales representativos ni controles de calidad de captura.

## Artefactos y reproducción

Los resultados nuevos están en `outputs/experiments/baseline_formal/baseline_con_aumento/security_threshold_calibration/`:

- `security_threshold_calibration.json`: política, candidatos y recomendaciones.
- `security_threshold_comparison.csv`: tabla consolidada de validation, test y peor stress.
- `stress_metrics_by_candidate.csv`: desglose por criterio y condición.
- `stress_predictions.csv`: scores reutilizables de las nueve condiciones; no contiene imágenes.
- `validation_threshold_search.csv`: barrido completo realizado solo con validation.
- cinco gráficos PNG livianos de curvas, candidatos y distribuciones.

Comando de reproducción en este entorno Windows, usando el dataset local compartido y el entorno virtual existente:

```powershell
$env:DUAL_ACCESS_DATA_DIR='C:\Users\carlo\OneDrive\Escritorio\Deep Learning\dual-access-control\data'
& '..\dual-access-control\.venv\Scripts\python.exe' -m src.evaluation.calibrate_security_thresholds --experiment-name baseline_formal/baseline_con_aumento --batch-size 32 --device auto
```

La primera ejecución genera scores de stress con el modelo; las siguientes reutilizan la caché. `--refresh-stress-predictions` fuerza una regeneración determinista.

## Limitaciones y revisión antes de producción

- Solo hay 250 parejas negativas por split; el FAR cambia en pasos de 0.4 puntos porcentuales y sus estimaciones tienen alta incertidumbre.
- Los 500 pares no representan intentos independientes a escala de despliegue ni todos los impostores posibles.
- Las pruebas de stress son sintéticas; no sustituyen capturas reales en el lugar de la demo.
- Debe verificarse en la web que la comparación sea `score >= 0.3128704727` y que RFID y rostro se combinen con una condición lógica AND.
- Deben definirse reintentos, bloqueo temporal, registro de eventos y un flujo seguro ante fallos del lector o de la cámara.
- Antes de producción se necesita una validation más amplia y representativa, con intervalos de confianza y, de ser posible, calibración por condiciones reales.
