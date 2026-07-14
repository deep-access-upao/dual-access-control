# Protocolo de dataset sin fuga

## Problema corregido

El flujo anterior creaba pares con todas las imágenes y recién después dividía esos pares. Por ello, una misma imagen, identidad y secuencia de video podía aparecer en entrenamiento, validación y prueba. Las métricas obtenidas con esos CSV no representan generalización y deben descartarse como evidencia experimental.

## Flujo operacional

El orden obligatorio es:

1. `build_manifest` registra cada imagen, persona, vista, video fuente, estado de calidad, dimensiones, tamaño y SHA-256.
2. `build_splits` asigna videos completos a `train`, `validation` o `test`.
3. `build_pairs` crea pares únicamente con imágenes del mismo split.
4. `audit_splits` verifica automáticamente que no exista fuga.

La unidad indivisible es el video fuente. Los frames consecutivos nunca se separan. Como el dataset actual tiene cuatro videos por persona, el protocolo predeterminado usa 50/25/25: dos videos para train, uno para validation y uno para test. Las identidades aparecen en los tres splits, pero las capturas y vistas asignadas son distintas. La semilla hace reproducible la selección.

```powershell
python -m src.dataset.build_manifest
python -m src.dataset.build_splits --seed 42
python -m src.dataset.build_pairs --seed 42 --overwrite
python -m src.dataset.audit_splits
```

Los CSV de pares se limitan con `--train-pairs`, `--validation-pairs` y `--test-pairs`. La configuración predeterminada crea 4,000/500/500 pares, balanceados 50/50. El muestreo reparte positivos entre personas y negativos entre combinaciones de personas para reducir el dominio de identidades con más frames.

El manifiesto y los CSV se generan localmente bajo `data/` y no se versionan porque contienen metadatos del dataset facial privado.

## Garantías de la auditoría

La auditoría falla con código distinto de cero ante imágenes o videos compartidos, hashes exactos cruzados, pares repetidos ignorando el orden A/B, imágenes de otro split dentro de un par, columnas faltantes o etiquetas inconsistentes. También informa imágenes y videos por split, distribución por persona y balance positivo/negativo.

Los hashes repetidos dentro de un mismo split se reportan, pero solo constituyen fuga si cruzan splits. `build_splits` rechaza de antemano hashes idénticos asociados a distintos videos o personas.

## Support set

`data/support_set/<person_id>/` admite una o más referencias con nombres como `frontal.jpg`, `frontal_02.jpg`, `left.jpg` o `right_02.jpg`. No se deben usar imágenes de train. Las queries de prueba permanecen fuera del support set y nunca deben convertirse en referencias.

```powershell
python -m src.dataset.validate_support_set
```

Cuando exista un manifiesto, el validador compara SHA-256 y falla si una referencia coincide con train. No copia ni duplica imágenes automáticamente.

## Pendiente

- Reentrenar el baseline desde cero con estos CSV y descartar las métricas anteriores.
- Definir el support set definitivo a partir de una sesión de enrolamiento independiente.
- Añadir un protocolo de identidades no vistas cuando haya suficientes personas. Se implementará asignando identidades completas a splits antes de generar pares; requiere al menos dos identidades por split para formar negativos y una muestra mayor para producir métricas estables.
