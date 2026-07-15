# Checklist de capturas y evidencias de la demo web

> Este checklist se concentra en la web. Para métricas, archivos, comandos y puntos orales de toda la entrega, usa el [checklist final de presentación](final_presentation_checklist.md).

## Preparación segura

- [ ] Usar nombre ficticio (`Usuario Demo 1`) y UID simulado (`01020304`).
- [ ] Obtener consentimiento para cualquier rostro mostrado en una exposición privada.
- [ ] Ocultar pestañas, notificaciones, usuarios del sistema y rutas locales privadas.
- [ ] No subir a Git fotos, capturas personales, bases SQLite, modelos ni datasets.
- [ ] Nombrar las evidencias por número de caso, sin nombres reales.

## Capturas obligatorias para informe o exposición

- [ ] **E01 — Inicio:** pantalla de inicio completa, con el texto `RFID simulado +
  verificación facial` y el threshold visible.
- [ ] **E02 — Registro:** formulario vacío de registro mostrando nombre, UID y selector
  de referencias; no mostrar rutas privadas del explorador de archivos.
- [ ] **E03 — Usuario registrado:** lista de usuarios con `Usuario Demo 1`, UID
  `01020304`, estado `Activo` y cantidad de referencias.
- [ ] **E04 — Referencias registradas:** vista de referencias con su cantidad o nombres
  internos; no mostrar ni publicar las imágenes faciales.
- [ ] **E05 — GRANTED:** resultado completo con `GRANTED`, `RFID_AND_FACE_OK`, nombre
  ficticio, score, threshold y referencias usadas.
- [ ] **E06 — UID desconocido:** resultado `DENIED`, motivo `RFID_UNKNOWN`, usuario
  `No identificado` y score `No calculado`.
- [ ] **E07 — Rostro incorrecto:** resultado `DENIED`, motivo `FACE_NO_MATCH`, score
  menor al threshold y cantidad de referencias.
- [ ] **E08 — Usuario inactivo:** lista de usuarios mostrando `Inactivo` y resultado
  `DENIED / USER_INACTIVE` (pueden ser dos capturas si mejora la legibilidad).
- [ ] **E09 — Historial:** tabla con los intentos anteriores y las columnas fecha, UID,
  usuario, score, threshold, decisión, motivo y referencias.
- [ ] **E10 — Configuración:** fragmento de `config/model_config.json` donde se vean
  únicamente `model_name`, `threshold`, `score_rule` y `access_rule`; evitar rutas
  locales o contenido del modelo.
- [ ] **E11 — Modelo validado:** terminal con el comando `check-model` y su resultado
  correcto; recortar cualquier ruta privada si aparece.
- [ ] **E12 — Tests:** terminal con `python -m unittest discover -s tests -v`, resumen
  `OK` y cantidad de tests.
- [ ] **E13 — RFID simulado:** pantalla inicial o formulario de verificación con una
  anotación visual: `UID escrito en web → usuario → rostro → decisión → historial`.

## Evidencias opcionales útiles

- [ ] `/health` en el navegador mostrando `status`, configuración, SQLite, modelo y
  threshold, sin rutas ni datos privados.
- [ ] Cámara web activa y mensaje `Captura lista para verificar`; usar solo si la
  persona autoriza aparecer en la evidencia.
- [ ] Alternativa de carga de archivo cuando la cámara no está disponible.
- [ ] Caso de iluminación difícil con condición, score y limitación anotados, sin
  publicar la imagen fuente.
- [ ] Ejecución del smoke test con todos los pasos `[OK]`.

## Revisión antes de entregar evidencias

- [ ] Cada captura tiene un identificador E01–E13 y una breve leyenda.
- [ ] Los valores de score y threshold son legibles en los casos faciales.
- [ ] El historial incluye tanto acceso concedido como rechazos por los tres motivos.
- [ ] No hay información sensible innecesaria ni contenido prohibido en el commit.
- [ ] La plantilla `integrated_demo_results_template.md` referencia las evidencias.
