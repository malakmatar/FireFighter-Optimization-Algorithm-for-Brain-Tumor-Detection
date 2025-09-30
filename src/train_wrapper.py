import gc
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import gc, tensorflow as tf
tf.keras.backend.clear_session()
gc.collect()

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    for _g in _gpus:
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except Exception:
            pass

def _build_vgg16_classifier(hparams, input_shape=(224, 224, 3), n_classes=4):
    dense_units = int(hparams["dense_units"])
    dropout = float(hparams["dropout_rate"])
    l2wd = float(hparams.get("l2_weight_decay", 0.0))
    unfreeze = int(hparams.get("unfrozen_blocks", 0))
    reg = regularizers.l2(l2wd) if l2wd > 0 else None

    base = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False
    if unfreeze >= 1:
        for layer in base.layers:
            if layer.name.startswith("block5"):
                layer.trainable = True
    if unfreeze >= 2:
        for layer in base.layers:
            if layer.name.startswith(("block4", "block5")):
                layer.trainable = True

    x = base.output
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg)(x)
    return models.Model(inputs=base.input, outputs=out)

def evaluate_model(hparams, train_X, train_y, val_X, val_y):
    lr = float(hparams["learning_rate"])
    bs = int(hparams["batch_size"])
    epochs = int(hparams.get("epochs", 3))
    unfreeze = int(hparams.get("unfrozen_blocks", 0))
    if unfreeze >= 2:
        lr *= 0.3

    model = None
    history = None
    try:
        K.clear_session()
        model = _build_vgg16_classifier(hparams)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cbs = [
            EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=1, min_lr=1e-6),
        ]
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=bs,
            callbacks=cbs,
            verbose=0,
        )
        return float(max(history.history["val_accuracy"]))
    finally:
        try:
            del history
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        K.clear_session()
        gc.collect()
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass
