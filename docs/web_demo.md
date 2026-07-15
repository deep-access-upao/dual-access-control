# Demo web con usuarios, support set y RFID simulado

## Objetivo

Esta demo local registra usuarios con un UID RFID y una o más fotos de referencia.
Al simular el UID y aportar una captura facial, consulta SQLite y usa el paquete
`src/inference` para comparar la captura contra el support set del usuario. La regla
es fija: **RFID conocido y activo AND rostro verificado = GRANTED**. Cualquier otro
caso produce `DENIED`.

No se entrena ni recalibra ningún modelo. La configuración usada es
`config/model_config.json`, con el modelo
`baseline_formal/baseline_con_aumento` y el threshold `0.3128704727`.

## Instalación

Desde la raíz del repositorio, en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

El archivo privado del modelo debe existir en la ruta indicada por
`config/model_config.json`. El modelo no se descarga ni se versiona.

## Inicialización y ejecución

La base y sus tablas se crean automáticamente al iniciar la aplicación:

```powershell
python -m src.web.app
```

También se puede ejecutar con:

```powershell
uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000`. La base queda en
`data/demo/dual_access_demo.sqlite3`; las referencias, en
`data/demo/reference_images/`. Ambos recursos están excluidos de Git.

## Registrar un usuario

1. Abre **Registrar**.
2. Escribe el nombre y un UID RFID único.
3. Selecciona una o más referencias JPG, JPEG, PNG o WEBP, de hasta 10 MB cada una.
4. Pulsa **Crear usuario**.
5. En **Usuarios**, confirma el UID, el estado activo y la cantidad de referencias.

Los UID se normalizan a mayúsculas y no distinguen entre mayúsculas y minúsculas.
Desde la lista se puede activar o desactivar un usuario y consultar sus referencias.

## Simular una verificación

En **Verificar**, pega el UID y aporta una captura. La carga de archivo siempre está
disponible. Como alternativa, abre **Usar cámara web**, concede permiso al navegador
y pulsa **Tomar captura**.

Para probar los resultados:

- **GRANTED:** usa el UID de un usuario activo y una captura facial que corresponda a
  alguna de sus referencias. El score máximo debe ser mayor o igual a `0.3128704727`.
- **DENIED por UID desconocido:** usa un UID no registrado. El motivo será
  `RFID_UNKNOWN` y el modelo no se ejecutará.
- **DENIED por usuario inactivo:** desactiva el usuario y usa su UID. El motivo será
  `USER_INACTIVE` y el modelo no se ejecutará.
- **DENIED por rostro incorrecto:** usa un UID activo y una captura de otra persona.
  Si el score máximo queda bajo el threshold, el motivo será `FACE_NO_MATCH`.

La respuesta muestra la decisión, el motivo, el score, el threshold, el usuario
identificado por RFID y la cantidad de referencias empleadas. La captura temporal se
elimina tras la evaluación.

## Integración de inferencia

La web carga el verificador mediante `FaceVerifier.from_config`, llama
`verify_against_references` con estrategia `max` y aplica `decide_access`. No duplica
la comparación facial ni cambia el threshold. Si el modelo o la inferencia fallan, el
sistema falla de forma segura con `DENIED` y motivo `INFERENCE_ERROR`.

## Historial

La pantalla **Historial** muestra los 200 eventos más recientes: fecha UTC, UID,
usuario, score, threshold, decisión, motivo y cantidad de referencias. Los eventos de
UID desconocido o usuario inactivo también se registran, con score vacío.

## Privacidad y limitaciones

- La demo es local y no incorpora autenticación; no debe exponerse directamente a
  Internet.
- La base, las fotos y las capturas son datos privados y no deben subirse a GitHub.
- Obtén consentimiento de cada persona y elimina sus datos cuando termine la demo.
- La iluminación difícil, el desenfoque, oclusiones y ángulos extremos pueden reducir
  la calidad de la verificación. Usa luz frontal uniforme y una captura nítida.
- La cámara requiere permiso del navegador y funciona en `localhost` o un contexto
  seguro. Si falla, usa la carga de archivo.
- Esta sesión simula el UID. La lectura RFID real y la respuesta física con ESP32 se
  integrarán en una sesión posterior.

## Comprobaciones recomendadas

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m src.inference.cli check-model --config config/model_config.json
```
