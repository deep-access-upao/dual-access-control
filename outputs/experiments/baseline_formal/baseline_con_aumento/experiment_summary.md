# Resumen — baseline formal con aumentos

- Dataset auditado sin fuga: 4000 pares train, 500 validation y 500 test; cada split está balanceado.
- Arquitectura: baseline siamés existente, sin cambios estructurales.
- Entrenamiento: aumentos solo en train; validation limpio; Adam `1e-4`; batch 64; semilla 42.
- Ejecución: 4 épocas completas de 6 máximas; mejor checkpoint en época 4 (`val_loss=0.323434`). La época 5 se detuvo por el límite de recursos CPU del entorno.
- Threshold: `0.0455392189`, calibrado únicamente con validation al maximizar F1 y desempatar por menor FAR/FRR.
- Validation: accuracy 0.9720, F1 0.9725, FAR 0.0480, FRR 0.0080, ROC AUC 0.9833.
- Test limpio: accuracy 0.9760, F1 0.9765, FAR 0.0440, FRR 0.0040, ROC AUC 0.9912.
- Robustez: poca luz es la condición más débil (accuracy 0.7680, FAR 0.3240); le siguen sobreexposición y bajo contraste.
- El modelo local está en `models/saved_model/baseline_formal/baseline_con_aumento.keras` y no se versiona por su tamaño (~50.6 MB).
