"""CNN architecture (TensorFlow / Keras) for 8x8 grayscale digit images."""
import keras
from keras import layers

from config import INPUT_SHAPE, NUM_CLASSES


def build_cnn(cfg: dict) -> keras.Model:
    """Build and compile the CNN described by a configuration dictionary.

    Architecture (sized for tiny 8x8 images - deliberately small):
        Input (8,8,1)
        Conv2D(32, 3x3, same, ReLU)     learns low-level strokes / edges
        MaxPooling2D(2x2)               8x8 -> 4x4, keeps strongest responses
        Conv2D(64, 3x3, same, ReLU)     combines strokes into curves / loops
        Flatten                         4*4*64 = 1024 features
        Dense(64, ReLU)                 learns feature combinations per class
        Dropout(rate)                   regularisation (rate 0.0 = disabled)
        Dense(10, softmax)              probabilities for digits 0-9
    """
    k = cfg["kernel_size"]
    inputs = keras.Input(shape=INPUT_SHAPE, name="digit_image")
    x = layers.Conv2D(cfg["conv1_filters"], (k, k), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(cfg["conv2_filters"], (k, k), padding="same", activation="relu", name="conv2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", name="dense1")(x)
    if cfg["dropout"] > 0:
        x = layers.Dropout(cfg["dropout"], name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"digit_cnn_{cfg['name']}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        # labels are integers 0-9, so the *sparse* categorical cross-entropy is
        # the correct loss (equivalent to one-hot + categorical cross-entropy).
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
