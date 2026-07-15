"""Variantes controladas del modelo siamés sin modificar el baseline histórico."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import INPUT_SHAPE
from src.models.siamese_network import EMBEDDING_SIZE, build_siamese_model

BASELINE_VARIANT = "baseline"
GAP_L2_COSINE_VARIANT = "gap_l2_cosine"
SUPPORTED_MODEL_VARIANTS = (BASELINE_VARIANT, GAP_L2_COSINE_VARIANT)

ARCHITECTURE_DESCRIPTIONS = {
    BASELINE_VARIANT: "CNN baseline con Flatten, embedding 128 y distancia L1",
    GAP_L2_COSINE_VARIANT: (
        "CNN propia con GlobalAveragePooling2D, embedding 128 normalizado L2 "
        "y similitud coseno"
    ),
}


def build_gap_l2_embedding_network(
    input_shape: tuple = INPUT_SHAPE,
    embedding_size: int = EMBEDDING_SIZE,
) -> tf.keras.Model:
    """Construye el encoder CNN de la variante GAP + L2."""
    inputs = layers.Input(shape=input_shape, name="embedding_input")

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="embedding_projection")(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(
        embedding_size, name="embedding_before_l2"
    )(x)
    normalized = layers.UnitNormalization(axis=1, name="embedding_l2")(embedding)

    return models.Model(
        inputs=inputs,
        outputs=normalized,
        name="gap_l2_embedding_network",
    )


def build_gap_l2_cosine_model(input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    """Compara embeddings unitarios con coseno y devuelve probabilidad [0, 1]."""
    embedding_network = build_gap_l2_embedding_network(input_shape)
    input_a = layers.Input(shape=input_shape, name="image_a")
    input_b = layers.Input(shape=input_shape, name="image_b")

    embedding_a = embedding_network(input_a)
    embedding_b = embedding_network(input_b)
    cosine = layers.Dot(
        axes=1,
        normalize=False,
        name="cosine_similarity",
    )([embedding_a, embedding_b])
    similarity = layers.Rescaling(
        scale=0.5,
        offset=0.5,
        name="similarity",
    )(cosine)

    return models.Model(
        inputs=[input_a, input_b],
        outputs=similarity,
        name="siamese_gap_l2_cosine",
    )


def build_siamese_variant(
    variant: str,
    input_shape: tuple = INPUT_SHAPE,
) -> tf.keras.Model:
    """Construye una variante conocida y falla de forma explícita ante typos."""
    if variant == BASELINE_VARIANT:
        return build_siamese_model(input_shape)
    if variant == GAP_L2_COSINE_VARIANT:
        return build_gap_l2_cosine_model(input_shape)
    choices = ", ".join(SUPPORTED_MODEL_VARIANTS)
    raise ValueError(f"Variante no soportada: {variant}. Opciones: {choices}")
