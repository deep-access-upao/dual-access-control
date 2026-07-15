# Checklist de evidencias de la demo web

Antes de tomar capturas, usa datos ficticios cuando sea posible y revisa que ninguna
imagen personal, UID real o ruta privada vaya a publicarse.

- [ ] Pantalla inicial en `http://127.0.0.1:8000` con threshold visible.
- [ ] Formulario de registro con nombre, UID y selector de referencias.
- [ ] Confirmación del usuario registrado y su UID simulado.
- [ ] Lista de usuarios con estado y cantidad de referencias.
- [ ] Vista de referencias registradas sin exponer las imágenes en Git.
- [ ] Resultado `GRANTED` con usuario, score, threshold y referencias usadas.
- [ ] Resultado `DENIED` por UID desconocido con motivo `RFID_UNKNOWN`.
- [ ] Resultado `DENIED` por rostro incorrecto con motivo `FACE_NO_MATCH`.
- [ ] Historial que contenga los eventos positivo y negativos.
- [ ] Opcional: cámara activa y captura preparada en el navegador.

## Datos que conviene anotar junto a las capturas

- Fecha y entorno local usado.
- Nombre ficticio del usuario de demo.
- UID RFID simulado.
- Cantidad de referencias.
- Score y decisión de la prueba positiva.
- Score y decisión de la prueba facial negativa.
- Confirmación de que la base y las imágenes permanecieron fuera del commit.
