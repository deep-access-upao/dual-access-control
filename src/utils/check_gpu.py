"""Diagnóstico breve de CPU, GPU y soporte CUDA de TensorFlow."""

from __future__ import annotations

import platform

import tensorflow as tf

from src.utils.tensorflow_runtime import configure_tensorflow_runtime


def _print_devices(label: str, devices: list[tf.config.PhysicalDevice]) -> None:
    print(f"{label} detectados: {len(devices)}")
    for index, device in enumerate(devices):
        print(f"  [{index}] {device.name}")


def _gpu_display_name(gpu: tf.config.PhysicalDevice) -> str:
    try:
        details = tf.config.experimental.get_device_details(gpu)
    except (RuntimeError, ValueError):
        details = {}
    return str(details.get("device_name") or gpu.name)


def main() -> None:
    print("=== Diagnóstico de TensorFlow y GPU ===")
    print(f"Python: {platform.python_version()} ({platform.platform()})")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Compilado con soporte CUDA: {tf.test.is_built_with_cuda()}")

    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    _print_devices("CPU", cpus)
    _print_devices("GPU", gpus)

    if not gpus:
        print("Memory growth: no aplica porque no se detectó GPU.")
        print(
            "Resultado: TensorFlow no detectó GPU; este entorno puede ejecutar el "
            "proyecto con CPU."
        )
        return

    for index, gpu in enumerate(gpus):
        print(f"  Nombre GPU [{index}]: {_gpu_display_name(gpu)}")

    try:
        runtime = configure_tensorflow_runtime("auto")
    except RuntimeError as error:
        print(f"Memory growth: no pudo activarse ({error})")
    else:
        enabled = set(runtime.memory_growth_devices)
        for gpu in gpus:
            status = "activado" if gpu.name in enabled else "no activado"
            print(f"Memory growth {gpu.name}: {status}")

    try:
        with tf.device("/GPU:0"):
            left = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            right = tf.constant([[2.0], [1.0]])
            result = tf.matmul(left, right)
        values = result.numpy().reshape(-1).tolist()
        executed_on_gpu = "GPU" in result.device.upper()
        print(f"Operación pequeña (matmul): {values}")
        print(f"Dispositivo de la operación: {result.device}")
        print(f"Operación ejecutada en GPU: {executed_on_gpu}")
    except (RuntimeError, tf.errors.OpError) as error:
        print(f"Operación en GPU: falló con un error claro: {error}")


if __name__ == "__main__":
    main()
