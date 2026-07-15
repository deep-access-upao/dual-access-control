# Matriz de cobertura de la rúbrica

La matriz distingue evidencia implementada de decisiones no priorizadas. No se atribuyen técnicas que no fueron ejecutadas.

| Ítem | Criterio | Evidencia en el proyecto | Archivo/Sección | Estado |
|---:|---|---|---|---|
| 1 | Definición del problema | Control de acceso de doble factor; riesgo de credencial prestada; verificación por similitud. | `README.md` — Problema y objetivo; `docs/final_methodology.md` §1 | Cubierto |
| 2 | Objetivos e indicadores | Objetivo medible y métricas accuracy, precision, recall, F1, FAR, FRR, ROC AUC y demo funcional. | `README.md` — Problema y objetivo; metodología §2 | Cubierto |
| 3 | EDA | Conteos de 11 personas, 44 videos, 1,702 detecciones, 1,544 utilizables, 158 rechazadas; notebook de preparación explora distribución por persona/vista. | `notebooks/01_dataset_preparation.ipynb`; metodología §3 | Cubierto |
| 4 | Limpieza y transformación | Detección/recorte, rechazo por calidad, RGB, 112x112x3, float32, normalización y pares positivos/negativos. | `src/preprocessing/`; `src/dataset/`; metodología §3.2 | Cubierto |
| 5 | División sin fuga | Split por video antes de pares; auditoría con cero imágenes, videos, pares y hashes cruzados. | `docs/dataset_protocol.md`; `src/dataset/audit_splits.py`; metodología §4 | Cubierto |
| 6 | Modelo baseline | CNN siamesa con pesos compartidos, embedding 128, distancia L1 y salida sigmoide. | `src/models/siamese_network.py`; metodología §6.1 | Cubierto |
| 7 | Configuración de entrenamiento | Adam, lr 1e-4, batch 64, 10 épocas, patience 5, seed 42, checkpoint por val_loss. | `docs/baseline_training_results.md`; `outputs/.../training_config.json`; metodología §5 | Cubierto |
| 8 | Control de experimentos | Config, historia, métricas, predicciones y plots por experimento; test excluido de entrenamiento/calibración. | `outputs/experiments/`; `src/training/train.py`; metodología §5 | Cubierto |
| 9 | Feature extraction | Baseline `Flatten`; evaluación controlada de GAP + embedding L2 + coseno; variante descartada por stress fotométrico. | `src/models/siamese_variants.py`; `docs/siamese_model_improvement.md`; metodología §6.2 | Cubierto mediante comparación |
| 10 | Fine-tuning | No aplicado: arquitectura propia y dataset pequeño. MobileNetV2 congelado y fine-tuning selectivo quedan como experimento futuro. | Metodología §6.3 y §12 | Justificado, no implementado |
| 11 | Teacher-student / distillation | No priorizada: faltó teacher validado y la prioridad fue fuga, robustez, threshold e integración. | Metodología §6.3 | Justificado, no implementado |
| 12 | Técnica complementaria | Augmentation solo train, stress determinista, `security_first`, inferencia modular y smoke test. | `docs/dataset_realism_and_augmentation.md`; `docs/security_threshold_calibration.md`; `src/inference/` | Cubierto |
| 13 | Métricas | Test final: accuracy 0.9660, precision 0.9794, recall 0.9520, F1 0.9655, FAR 0.0200, FRR 0.0480, AUC 0.9838 y matriz. | Metodología §8.1; outputs del baseline final | Cubierto |
| 14 | Comparación de experimentos | Con/sin augmentation; max_f1/security_first; baseline/GAP+L2+coseno; limpio/stress. | `docs/baseline_comparison_results.md`; `docs/siamese_model_improvement.md`; notebook 99 | Cubierto |
| 15 | Análisis de errores | Poca luz, sobreexposición y bajo contraste elevan FRR; mitigación y mejoras propuestas. | Metodología §9; notebook 99 — Análisis de errores | Cubierto |
| 16 | Mejoras implementadas | Corrección de fuga, augmentation, calibración, variante, inferencia, web y pruebas integrales. | Metodología §11; historial documental y código | Cubierto |
| 17 | Limitaciones y mejoras futuras | Dataset pequeño, RFID simulado, sin externos/deploy/validation-stress/backbone final; plan MobileNetV2 y más datos. | `README.md` — Limitaciones; metodología §12 | Cubierto |
| 18 | Organización y reproducibilidad | Requirements, seeds, rutas, config, comandos, CLI, tests y script de regeneración controlada. | `docs/final_usage_guide.md`; `config/model_config.json`; `scripts/` | Cubierto |
| 19 | Presentación técnica | Notebook liviano de 18 secciones, README principal y checklist de exposición/evidencias. | `notebooks/99_project_summary.ipynb`; `docs/final_presentation_checklist.md` | Cubierto |
| 20 | Ética y uso responsable | Consentimiento, privacidad, no versionar biometría, sesgos, FAR/FRR y alcance académico. | `README.md` — Ética; metodología §14; notebook 99 | Cubierto |

## Correspondencia por bloque

- Problema y objetivos: ítems 1–2.
- Datos, EDA y transformación: ítems 3–5.
- Diseño y técnicas de Deep Learning: ítems 6–12.
- Evaluación y análisis crítico: ítems 13–17.
- Reproducibilidad y comunicación: ítems 18–19.
- Ética: ítem 20.

Los ítems 10 y 11 están documentados con honestidad como técnicas no implementadas. La alternativa de feature extraction sí fue implementada y evaluada, pero no seleccionada. Esta distinción evita presentar trabajo futuro como resultado experimental.
