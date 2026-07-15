# Plantilla de resultados de pruebas integrales

## Identificación de la ejecución

| Campo | Valor |
|---|---|
| Responsable | |
| Fecha y hora | |
| Rama/commit evaluado | |
| URL local | `http://127.0.0.1:8000` |
| Navegador y versión | |
| Sistema operativo | |
| Modelo usado | `baseline_formal/baseline_con_aumento` |
| Threshold | `0.3128704727` |
| Resultado de `/health` | |
| Resultado de `check-model` | |
| Resultado de tests automáticos | |

## Resultados por caso

Usar una fila por ejecución. En **Evidencia/captura**, registrar un identificador o
nombre de archivo del informe; no incrustar fotos faciales ni rutas privadas en Git.

| Fecha | Navegador | Sistema operativo | Modelo usado | Threshold | Caso de prueba | Datos usados | Resultado esperado | Resultado obtenido | Evidencia/captura | Observaciones |
|---|---|---|---|---:|---|---|---|---|---|---|
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 1: registro | | Usuario y referencias creados | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 2: GRANTED | | `GRANTED`, score ≥ threshold | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 3: UID desconocido | | `DENIED / RFID_UNKNOWN`, sin inferencia | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 4: rostro incorrecto | | `DENIED / FACE_NO_MATCH`, score < threshold | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 5: usuario inactivo | | `DENIED / USER_INACTIVE`, sin inferencia | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 6: historial | | Todos los intentos registrados | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 7: cámara web | | Captura enviada o carga alternativa validada | | | |
| | | | `baseline_formal/baseline_con_aumento` | `0.3128704727` | Caso 8: iluminación difícil | | Respuesta estable y limitación documentada | | | |

## Pruebas exitosas

- Casos aprobados:
- Comportamientos confirmados:
- Evidencias asociadas:

## Fallos encontrados

| ID | Caso | Descripción | Severidad | Pasos para reproducir | Evidencia | Estado |
|---|---|---|---|---|---|---|
| | | | | | | |

## Acciones correctivas

| ID de fallo | Acción propuesta o aplicada | Responsable | Fecha objetivo | Resultado de revalidación |
|---|---|---|---|---|
| | | | | |

## Conclusión de validación

- Resultado global: **APROBADO / APROBADO CON OBSERVACIONES / NO APROBADO**
- Cobertura ejecutada:
- Riesgos o limitaciones remanentes:
- Confirmación de que no se entrenó ni recalibró el modelo:
- Confirmación de que no se versionaron datos, imágenes, bases ni modelos:
- Recomendación antes del merge:
