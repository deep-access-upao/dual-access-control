"""Configuración segura del dispositivo usado por TensorFlow."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


SUPPORTED_DEVICES = ("auto", "cpu", "gpu")


@dataclass(frozen=True)
class TensorFlowRuntime:
    """Resumen de la configuración aplicada al runtime de TensorFlow."""

    requested_device: str
    selected_device: str
    cpu_devices: tuple[str, ...]
    gpu_devices: tuple[str, ...]
    memory_growth_devices: tuple[str, ...]


def _physical_device_names(device_type: str) -> tuple[str, ...]:
    return tuple(
        physical_device.name
        for physical_device in tf.config.list_physical_devices(device_type)
    )


def configure_tensorflow_runtime(device: str = "auto") -> TensorFlowRuntime:
    """Configura CPU/GPU antes de crear tensores o modelos.

    ``auto`` usa GPU cuando TensorFlow la detecta y cae a CPU en caso contrario.
    ``cpu`` oculta las GPU al runtime. ``gpu`` exige al menos una GPU visible.
    Cuando se usa GPU se activa memory growth para evitar que TensorFlow reserve
    toda la memoria disponible al iniciar.
    """
    requested_device = device.strip().lower()
    if requested_device not in SUPPORTED_DEVICES:
        choices = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(
            f"Dispositivo no soportado: {device!r}. Opciones válidas: {choices}."
        )

    cpu_names = _physical_device_names("CPU")
    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_names = tuple(gpu.name for gpu in physical_gpus)

    if requested_device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            raise RuntimeError(
                "No se pudo forzar CPU porque TensorFlow ya inicializó el runtime. "
                "Configura el dispositivo antes de crear tensores o cargar modelos."
            ) from error
        print("TensorFlow: se forzó el uso de CPU (--device cpu).")
        return TensorFlowRuntime(
            requested_device=requested_device,
            selected_device="cpu",
            cpu_devices=cpu_names,
            gpu_devices=gpu_names,
            memory_growth_devices=(),
        )

    if not physical_gpus:
        if requested_device == "gpu":
            raise RuntimeError(
                "Se exigió GPU (--device gpu), pero TensorFlow no detectó ninguna. "
                "Ejecuta 'python -m src.utils.check_gpu' para diagnosticar el entorno."
            )
        print("TensorFlow: no se detectó GPU; se usará CPU (--device auto).")
        return TensorFlowRuntime(
            requested_device=requested_device,
            selected_device="cpu",
            cpu_devices=cpu_names,
            gpu_devices=(),
            memory_growth_devices=(),
        )

    memory_growth_devices = []
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            memory_growth_devices.append(gpu.name)
        except RuntimeError as error:
            try:
                already_enabled = tf.config.experimental.get_memory_growth(gpu)
            except (RuntimeError, ValueError):
                already_enabled = False
            if not already_enabled:
                raise RuntimeError(
                    f"No se pudo activar memory growth para {gpu.name}. "
                    "Configura TensorFlow antes de crear tensores o cargar modelos."
                ) from error
            memory_growth_devices.append(gpu.name)

    print(
        "TensorFlow: se usará GPU; memory growth activado en "
        f"{len(memory_growth_devices)} dispositivo(s)."
    )
    for gpu_name in gpu_names:
        print(f"  GPU: {gpu_name}")
    return TensorFlowRuntime(
        requested_device=requested_device,
        selected_device="gpu",
        cpu_devices=cpu_names,
        gpu_devices=gpu_names,
        memory_growth_devices=tuple(memory_growth_devices),
    )
