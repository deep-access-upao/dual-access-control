"""Errores públicos del paquete de inferencia."""


class InferencePackageError(RuntimeError):
    """Error base con un mensaje apto para CLI, backend o web."""


class InvalidConfigError(InferencePackageError):
    """La configuración es inexistente, ilegible o incompatible."""


class ModelNotFoundError(InferencePackageError):
    """No se encontró el artefacto del modelo."""


class ModelLoadError(InferencePackageError):
    """Keras no pudo cargar o validar el modelo."""


class ImageNotFoundError(InferencePackageError):
    """No se encontró una imagen de entrada."""


class InvalidImageError(InferencePackageError):
    """La imagen no se pudo decodificar o no tiene el shape esperado."""


class InferenceError(InferencePackageError):
    """El modelo no pudo producir un score de similitud válido."""
