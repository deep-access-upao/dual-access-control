# Realismo del dataset, aumentos y pruebas de estrés

## Objetivo metodológico

El dataset original contiene rostros válidos para entrenar el prototipo, pero no cubre necesariamente cambios habituales de una implementación real: iluminación, calidad de cámara, encuadre, oclusiones o apariencia. El pipeline incorpora variaciones moderadas para reducir esa brecha sin alterar el protocolo sin fuga:

`manifiesto → split por video → pares dentro de cada split → auditoría sin fuga`

Los aumentos se calculan en memoria y no modifican ni duplican las imágenes originales.

## Separación entre entrenamiento y evaluación

- `train`: usa aumentos aleatorios con `augment=True` de forma predeterminada.
- `validation`: usa siempre `augment=False` y no se mezcla aleatoriamente.
- `test`: usa siempre `augment=False` y no se mezcla aleatoriamente.
- Las pruebas de estrés son evaluaciones adicionales. No reemplazan ni contaminan el test limpio.

Cada rama del par siamés recibe una semilla diferente, por lo que `image_a` e `image_b` pueden experimentar variaciones independientes. La semilla base hace reproducible una ejecución del loader.

## Aumentos aplicados a `train`

El módulo `src/dataset/augmentations.py` mantiene el tamaño `112 × 112 × 3` y aplica variaciones moderadas:

- rotación de hasta aproximadamente 7 grados;
- traslación de hasta 4 % por eje;
- zoom leve;
- brillo, contraste y gamma;
- ruido gaussiano suave;
- desenfoque ligero;
- degradación leve de resolución, como aproximación a compresión/calidad de cámara;
- oclusión pequeña tipo *cutout*.

La geometría y la iluminación cambian suavemente en cada muestra. Ruido, blur, degradación y cutout se activan con probabilidades acotadas para no destruir rasgos de identidad.

Uso previsto en el entrenamiento futuro:

```python
from src.dataset.dataloader import get_train_dataset, get_val_dataset

train_dataset = get_train_dataset(batch_size=32, augment=True, seed=42)
val_dataset = get_val_dataset(batch_size=32)  # siempre limpio
```

El script `src/training/train.py` ya activa los aumentos. Para una comparación controlada sin ellos se podrá usar `--no-augmentation`; no es necesario ni recomendable cambiar validation.

## Pruebas de estrés deterministas

`src/evaluation/stress_tests.py` ofrece condiciones independientes y reproducibles:

- poca luz;
- sobreexposición;
- contraste bajo;
- ruido;
- blur;
- rotación y cambio de encuadre;
- oclusión parcial;
- lentes sintéticos simples;
- sombra de barba sintética simple.

Ejemplo:

```python
from src.dataset.dataloader import load_image
from src.evaluation.stress_tests import apply_stress_condition

image = load_image("ruta/a/imagen.jpg")
low_light = apply_stress_condition(image, "low_light", seed=[2026, 1])
```

Estas transformaciones no escriben archivos. Si se exportan previews manualmente, deben guardarse en `outputs/augmentation_preview/` o `outputs/stress_tests_preview/`, carpetas ignoradas por Git.

## Cómo revisar el preview

1. Ejecutar `notebooks/01_dataset_preparation.ipynb` para preparar y auditar el dataset.
2. Ejecutar `notebooks/02_pair_generation_preview.ipynb` para generar/revisar los pares.
3. Ejecutar `notebooks/02b_augmentation_preview.ipynb` para inspeccionar aumentos y estrés sin entrenar.
4. Continuar con `notebooks/03_siamese_model_summary.ipynb` y los notebooks posteriores cuando corresponda.

El notebook `02b` toma una ruta del CSV de entrenamiento, construye todo en memoria y no guarda salidas pesadas.

## Limitación importante

Los lentes y la barba simulados son pruebas de estrés sintéticas muy simples. Sirven para detectar sensibilidad obvia del pipeline, pero no sustituyen fotografías reales: no modelan correctamente reflejos, refracción, monturas variadas, textura de vello, crecimiento real ni cambios faciales correlacionados.

Antes de presentar resultados finales se recomienda capturar una segunda sesión real por persona, manteniéndola separada según el protocolo por video. Esa sesión debería incluir cambios de luz y fondo, lentes cuando sea posible, distancia/encuadre y variaciones reales de apariencia. El rendimiento debe reportarse por separado para test limpio, estrés sintético y sesión real externa.
