# Resumen — baseline formal con aumentos

- Dataset auditado sin fuga: 4000 pares train, 500 validation y 500 test; cada split está balanceado.
- Arquitectura: baseline siamés existente, sin cambios estructurales.
- Entrenamiento: aumentos solo en train; validation limpio; Adam `1e-4`; batch 64; semilla 42.
- Ejecución: 10 de 10 épocas completas. El entrenamiento se reanudó desde checkpoints completos conservando el estado de Adam.
- Mejor checkpoint: época 8, `val_loss=0.2569648027`; épocas 9 y 10 no lo mejoraron.
- Threshold: `0.0187880173`, calibrado únicamente con validation al maximizar F1 y desempatar por menor FAR/FRR.
- Validation: accuracy 0.9840, F1 0.9841, FAR 0.0240, FRR 0.0080, ROC AUC 0.9853.
- Test limpio: accuracy 0.9800, F1 0.9803, FAR 0.0360, FRR 0.0040, ROC AUC 0.9838.
- Robustez: poca luz sigue siendo la condición más débil (accuracy 0.8440, FAR 0.2240); le siguen bajo contraste y sobreexposición.
- El modelo local está en `models/saved_model/baseline_formal/baseline_con_aumento.keras` y no se versiona por su tamaño (~50.6 MB).
