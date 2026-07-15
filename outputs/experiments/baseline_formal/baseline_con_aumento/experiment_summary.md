# Resumen — baseline formal con aumentos

- Dataset auditado sin fuga: 4000 pares train, 500 validation y 500 test; cada split está balanceado.
- Arquitectura: baseline siamés existente, sin cambios estructurales.
- Entrenamiento: aumentos solo en train; validation limpio; Adam `1e-4`; batch 64; semilla 42.
- Ejecución: 6 de 6 épocas completas. Las épocas 5 y 6 se reanudaron desde el checkpoint de época 4 conservando el estado de Adam.
- Mejor checkpoint: época 6, `val_loss=0.2735352516`.
- Threshold: `0.0157270245`, calibrado únicamente con validation al maximizar F1 y desempatar por menor FAR/FRR.
- Validation: accuracy 0.9820, F1 0.9822, FAR 0.0280, FRR 0.0080, ROC AUC 0.9880.
- Test limpio: accuracy 0.9740, F1 0.9745, FAR 0.0440, FRR 0.0080, ROC AUC 0.9883.
- Robustez: poca luz es la condición más débil (accuracy 0.8060, FAR 0.2680); le siguen bajo contraste y sobreexposición.
- El modelo local está en `models/saved_model/baseline_formal/baseline_con_aumento.keras` y no se versiona por su tamaño (~50.6 MB).
