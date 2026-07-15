# Plan de pruebas integrales de la demo web

> Este documento conserva el detalle operativo de los casos. La selección de evidencias y el guion de exposición están resumidos en el [checklist final de presentación](final_presentation_checklist.md).

## Objetivo y alcance

Validar de extremo a extremo el registro, el RFID simulado, la verificación facial,
la decisión de acceso y el historial de la demo FastAPI. Este plan no entrena el
modelo, no recalibra el threshold y no requiere ESP32. La configuración bajo prueba
es `baseline_formal/baseline_con_aumento`, con threshold `0.3128704727` y regla
`RFID conocido AND rostro verificado => GRANTED`.

## Entorno y preparación

1. Trabajar desde la raíz del repositorio en Windows nativo.
2. Confirmar que el modelo privado existe mediante:
   `python -m src.inference.cli check-model --config config/model_config.json`.
3. Iniciar la aplicación con `python -m src.web.app` y abrir
   `http://127.0.0.1:8000`.
4. Usar una base local de demo sin datos previos o elegir un UID nuevo.
5. Mantener fuera de Git la base SQLite, las referencias, las capturas y toda
   evidencia facial.
6. Disponer localmente, con consentimiento, de al menos dos personas diferentes:
   referencias y captura positiva de `user_001`, y captura negativa de `user_002`.

Anotar navegador, versión, sistema operativo, fecha y resultado de `/health` antes
de ejecutar los casos. Los scores dependen de las imágenes; conservar solo la cifra
y la evidencia autorizada, no las fotos, en los documentos versionados.

## Casos de prueba

### Caso 1. Registro correcto de usuario

**Datos:** nombre `Usuario Demo 1`, UID `01020304` y una o más imágenes locales de
referencia de `user_001`.

**Pasos:**

1. Abrir **Registrar**.
2. Introducir el nombre y UID indicados.
3. Seleccionar una o más referencias y crear el usuario.
4. Abrir **Usuarios** y luego la vista de referencias del usuario.

**Resultado esperado:** aparece la confirmación de registro; el usuario queda activo,
el UID se muestra como `01020304` y la cantidad de referencias coincide con los
archivos seleccionados. Las referencias quedan en almacenamiento local privado.

### Caso 2. Acceso GRANTED

**Precondición:** Caso 1 completado y usuario activo.

**Datos:** UID `01020304` y una captura diferente del mismo `user_001`.

**Pasos:** abrir **Verificar**, ingresar el UID, cargar o tomar la captura y evaluar.

**Resultado esperado:** `access_decision = GRANTED`, `reason = RFID_AND_FACE_OK`,
`face_match = true` y `score >= 0.3128704727`. La pantalla muestra Usuario Demo 1,
score, threshold y número de referencias usadas; se crea un evento en el historial.

### Caso 3. DENIED por UID desconocido

**Datos:** UID `99999999` y cualquier captura válida.

**Pasos:** ejecutar una verificación con esos datos y revisar el historial.

**Resultado esperado:** `access_decision = DENIED`, `reason = RFID_UNKNOWN`, score
`No calculado` y cero referencias. No se ejecuta inferencia facial porque el UID no
existe; el intento sí queda registrado.

### Caso 4. DENIED por rostro incorrecto

**Precondición:** Usuario Demo 1 activo.

**Datos:** UID `01020304` y captura de otra persona, por ejemplo `user_002`.

**Pasos:** ejecutar una verificación con esos datos y revisar el historial.

**Resultado esperado:** `access_decision = DENIED`, `reason = FACE_NO_MATCH`,
`face_match = false` y `score < 0.3128704727`. Se muestran el usuario asociado al
RFID, score, threshold y referencias usadas; el intento queda registrado.

### Caso 5. DENIED por usuario inactivo

**Precondición:** Usuario Demo 1 registrado.

**Pasos:**

1. Abrir **Usuarios** y desactivar Usuario Demo 1.
2. Verificar el UID `01020304` con una captura válida del mismo usuario.
3. Revisar el historial y reactivar el usuario al terminar.

**Resultado esperado:** el estado visible es `Inactivo`; la verificación devuelve
`access_decision = DENIED`, `reason = USER_INACTIVE`, score `No calculado` y cero
referencias usadas. No se ejecuta inferencia y el intento queda registrado.

### Caso 6. Historial de eventos

**Precondición:** Casos 2 a 5 ejecutados.

**Pasos:** abrir **Historial** y localizar cada intento por UID, decisión y motivo.

**Resultado esperado:** todos los intentos aparecen en orden descendente e incluyen
fecha UTC, UID, usuario cuando existe, score cuando se calculó, threshold, decisión,
motivo y cantidad de referencias. Deben estar presentes `GRANTED`, `RFID_UNKNOWN`,
`FACE_NO_MATCH` y `USER_INACTIVE`.

### Caso 7. Cámara web

**Pasos:**

1. En **Verificar**, desplegar **Usar cámara web**.
2. Activar la cámara y conceder permiso al navegador.
3. Tomar una captura, confirmar el mensaje `Captura lista para verificar` y enviarla.
4. Repetir mediante carga de archivo si no hay cámara o se deniega el permiso.

**Resultado esperado:** la captura de cámara se adjunta y puede enviarse. Si
`getUserMedia` no está disponible, la interfaz informa el problema y la carga de
archivo continúa funcionando como alternativa. La decisión facial depende de la
identidad y calidad de la captura.

### Caso 8. Iluminación difícil

**Datos:** captura con baja luz, desenfoque u otra degradación controlada y consentida.

**Pasos:** verificar el UID correcto con la imagen degradada y anotar score, decisión y
condiciones de captura.

**Resultado esperado:** la aplicación responde sin error y registra el evento. Puede
rechazar el rostro si el score queda bajo el threshold; documentar el resultado como
limitación conocida, no modificar el threshold para forzar aceptación.

## Criterios de cierre

- `/health` informa `status = OK`, configuración encontrada, SQLite disponible,
  nombre del modelo y threshold correctos.
- Los ocho casos tienen resultado y evidencia o una justificación documentada.
- Todo intento de acceso probado aparece en el historial.
- UID desconocido y usuario inactivo no invocan inferencia facial.
- No se cambió el modelo, threshold, dataset, splits ni pares.
- No se versionaron bases, modelos, imágenes, videos, capturas ni datos personales.
