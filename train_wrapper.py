"""
Model builder + evaluate_model wrapper (VGG16 backbone)

This script performs the following tasks:
1. Constructs a VGG16-based classifier with optional fine-tuning depth
2. Compiles with Adam optimizer and categorical crossentropy
3. Trains with early stopping and LR scheduling, returns best val_accuracy

Usage:
    Pipelined with ffo.py to evaluate all of the agents performances with diffrient hyperparameters
"""


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=1, min_lr=1e-6),  # gently lower LR when plateaus are detected

def evaluate_model(
    #Train a model with provided hparams and data, return max validation accuracy as float
hparams, train_X, train_y, val_X, val_y):
    dense_units = int(hparams["dense_units"])
    dropout = float(hparams["dropout_rate"])
    lr = float(hparams["learning_rate"])
    bs = int(hparams["batch_size"])
    l2wd = float(hparams.get("l2_weight_decay", 0.0))
    unfreeze = int(hparams.get("unfrozen_blocks", 0))
    EPOCHS = int(hparams.get("epochs", 3))  # fixed per mode by ffo.py

    base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False

    # Unfreeze depth if requested
    if unfreeze >= 1:
        for layer in base.layers:
            if layer.name.startswith("block5"):
                layer.trainable = True
    if unfreeze >= 2:
        for layer in base.layers:
            if layer.name.startswith(("block4", "block5")):
                layer.trainable = True
        lr *= 0.3  # safer LR when fine-tuning deeper

    reg = regularizers.l2(l2wd) if l2wd > 0 else None

    x = base.output
    x = layers.Flatten()(x)  # simple head to keep comparisons fair
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(4, activation="softmax", kernel_regularizer=reg)(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",   # <-- use sparse for integer labels
        metrics=["accuracy"]
    )

    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=1, min_lr=1e-6),  # gently lower LR when plateaus are detected
    ]

    hist = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=EPOCHS,
        batch_size=bs,
        callbacks=cbs,
        verbose=0
    )

    return float(max(hist.history["val_accuracy"]))