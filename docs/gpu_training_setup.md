# Entrenamiento con GPU mediante WSL2

## Ruta recomendada

TensorFlow 2.10 fue la última versión con soporte oficial para GPU NVIDIA en Windows nativo. Desde TensorFlow 2.11, la ruta recomendada para una GPU NVIDIA en Windows es ejecutar TensorFlow dentro de WSL2. La instalación oficial actual para Linux/WSL2 incluye las dependencias CUDA necesarias mediante `tensorflow[and-cuda]`.

Referencias oficiales:

- [Instalación de TensorFlow con pip](https://www.tensorflow.org/install/pip)
- [Instalación de WSL](https://learn.microsoft.com/windows/wsl/install)
- [CUDA en WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

No instales un driver NVIDIA para Linux dentro de WSL. WSL utiliza el driver NVIDIA instalado en el host Windows. La instalación y actualización de WSL, Ubuntu y el driver de Windows son tareas manuales del administrador del equipo; este repositorio no las realiza.

## 1. Comprobaciones e instalación en Windows

Abre PowerShell y comprueba primero que Windows reconoce la GPU y que la distribución usa WSL2:

```powershell
nvidia-smi
wsl --status
wsl --list --verbose
```

Si WSL todavía no está instalado, abre PowerShell como administrador y ejecuta:

```powershell
wsl --install -d Ubuntu
```

Reinicia Windows si se solicita, abre Ubuntu y crea el usuario Linux. Para actualizar una instalación de WSL existente:

```powershell
wsl --update
wsl --shutdown
```

Si `nvidia-smi` falla en Windows, instala o actualiza manualmente el driver NVIDIA compatible antes de continuar. No instales drivers desde este proyecto.

## 2. Preparar Ubuntu en WSL2

Dentro de Ubuntu, confirma que la GPU se expone a WSL:

```bash
nvidia-smi
```

Trabaja en el sistema de archivos Linux. Evita entrenar desde OneDrive o desde rutas montadas como `/mnt/c`, porque el acceso intensivo a muchos archivos suele ser más lento. Una ubicación recomendada es `~/proyectos`:

```bash
mkdir -p ~/proyectos
cd ~/proyectos
git clone https://github.com/deep-access-upao/dual-access-control.git
cd dual-access-control
git checkout soporte-entrenamiento-gpu
```

Instala las herramientas básicas de Python si Ubuntu aún no las tiene:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

Crea y activa un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instala las dependencias del proyecto y el extra oficial de TensorFlow para CUDA:

```bash
python -m pip install -r requirements.txt
python -m pip install "tensorflow[and-cuda]"
```

## 3. Verificar TensorFlow y la GPU

Desde la raíz del repositorio y con el entorno virtual activo:

```bash
python -m src.utils.check_gpu
```

El diagnóstico muestra las versiones de Python y TensorFlow, soporte CUDA, dispositivos físicos, estado de `memory_growth` y una multiplicación pequeña en `/GPU:0`. No ejecuta entrenamiento. Si no existe una GPU visible, termina normalmente e indica que el entorno usará CPU.

## 4. Seleccionar el dispositivo

Los comandos anteriores siguen funcionando sin cambios porque `--device` usa `auto` por defecto.

```bash
# GPU si está disponible; CPU en caso contrario
python -m src.training.train --device auto

# Oculta las GPU y fuerza CPU
python -m src.training.train --device cpu

# Exige una GPU; falla claramente si TensorFlow no detecta ninguna
python -m src.training.train --device gpu
```

El mismo argumento está disponible en las evaluaciones que cargan modelos:

```bash
python -m src.evaluation.evaluate --mode calibrate --device auto
python -m src.evaluation.evaluate_stress --device gpu
```

Cuando se selecciona GPU, el proyecto activa `memory_growth` antes de cargar o construir el modelo. Así TensorFlow aumenta el uso de memoria según lo necesita en lugar de reservar toda la VRAM al inicio.

## 5. Si TensorFlow muestra `[]` para las GPU

Comprueba en este orden:

1. `nvidia-smi` funciona en Windows.
2. `wsl --list --verbose` muestra la distribución en versión 2.
3. `nvidia-smi` funciona dentro de Ubuntu.
4. El entorno virtual correcto está activo y `python -m pip show tensorflow` apunta a ese entorno.
5. `python -m src.utils.check_gpu` informa que TensorFlow fue compilado con CUDA.

Después actualiza WSL desde PowerShell con `wsl --update`, ejecuta `wsl --shutdown` y vuelve a abrir Ubuntu. Si la GPU sigue sin aparecer, reinstala las dependencias dentro de un entorno virtual limpio y revisa la tabla de versiones de Python admitidas en la guía oficial de TensorFlow. No instales el driver Linux de NVIDIA dentro de WSL ni mezcles paquetes globales con el entorno virtual.
