import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses

from src.config import (
    INPUT_SHAPE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)

EMBEDDING_SIZE = 128


def build_embedding_network(input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="embedding_input")

    # Bloque 1: 112 -> 56
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Bloque 2: 56 -> 28
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Bloque 3: 28 -> 14
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Bloque 4: 14 -> 7
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Capa final del embedding sin activación: vector denso de tamaño fijo
    embedding = layers.Dense(EMBEDDING_SIZE, name="embedding")(x)

    return models.Model(inputs=inputs, outputs=embedding, name="embedding_network")


def l1_distance(embeddings: list) -> tf.Tensor:
    embedding_a, embedding_b = embeddings
    return tf.abs(embedding_a - embedding_b)


def build_siamese_model(input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    embedding_network = build_embedding_network(input_shape)

    input_a = layers.Input(shape=input_shape, name="image_a")
    input_b = layers.Input(shape=input_shape, name="image_b")

    # Pesos compartidos: ambas imágenes pasan por la misma red de embedding
    embedding_a = embedding_network(input_a)
    embedding_b = embedding_network(input_b)

    distance = layers.Lambda(l1_distance, name="l1_distance")([embedding_a, embedding_b])

    similarity = layers.Dense(1, activation="sigmoid", name="similarity")(distance)

    return models.Model(
        inputs=[input_a, input_b],
        outputs=similarity,
        name="siamese_network",
    )


def compile_siamese_model(
    model: tf.keras.Model,
    learning_rate: float = 0.0001,
) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="binary_accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )
    return model


def main() -> None:
    print("=== Red Siamesa — Construcción ===\n")

    embedding_network = build_embedding_network()
    siamese_model = build_siamese_model()
    siamese_model = compile_siamese_model(siamese_model)

    print("--- Resumen: Red de Embedding ---")
    embedding_network.summary()

    print("\n--- Resumen: Modelo Siamés ---")
    siamese_model.summary()

    # Prueba rápida con tensores aleatorios para validar el grafo
    dummy_a = tf.random.uniform((2, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    dummy_b = tf.random.uniform((2, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    dummy_output = siamese_model([dummy_a, dummy_b], training=False)

    print("\n--- Prueba forward con tensores aleatorios ---")
    print(f"  forma entrada a    : {dummy_a.shape}")
    print(f"  forma entrada b    : {dummy_b.shape}")
    print(f"  forma salida       : {dummy_output.shape}")
    print(f"  valores de salida  : {dummy_output.numpy().flatten()}")
    print("\nRed Siamesa OK.")


if __name__ == "__main__":
    main()
