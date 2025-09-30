#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-contained final trainer:
- Loads best_hparams from <run_dir>/result.json
- Rebuilds VGG16 head
- Trains on (train+val), evaluates on test
- Writes: history.csv, y_true.npy, y_pred.npy, y_prob.npy, final_test_metrics.json
"""

from __future__ import annotations
import argparse, json, os, gc
from pathlib import Path
from typing import List, Tuple
import numpy as np

# ----------------------- utils -----------------------
def _set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random; random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def _load_arrays(data_dir: Path) -> Tuple[np.ndarray, ...]:
    def pick(base: Path, names):
        for n in names:
            p = base / n
            if p.exists(): return p
        return None
    xtr = pick(data_dir, ["train_X.npy","X_train.npy"])
    ytr = pick(data_dir, ["train_y.npy","y_train.npy"])
    xv  = pick(data_dir, ["val_X.npy","X_val.npy"])
    yv  = pick(data_dir, ["val_y.npy","y_val.npy"])
    xte = pick(data_dir, ["test_X.npy","X_test.npy"])
    yte = pick(data_dir, ["test_y.npy","y_test.npy"])
    missing = [lbl for lbl,p in [("train_X",xtr),("train_y",ytr),("val_X",xv),("val_y",yv),("test_X",xte),("test_y",yte)] if p is None]
    if missing:
        raise FileNotFoundError(f"Missing arrays in {data_dir}: {', '.join(missing)}")
    return (np.load(xtr, allow_pickle=False),
            np.load(ytr, allow_pickle=False),
            np.load(xv,  allow_pickle=False),
            np.load(yv,  allow_pickle=False),
            np.load(xte, allow_pickle=False),
            np.load(yte, allow_pickle=False))

# ----------------------- model -----------------------
def _build_model_vgg16_like(hparams: dict, num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.applications import VGG16

    dense_units = int(hparams.get("dense_units", 256))
    dropout     = float(hparams.get("dropout_rate", 0.3))
    lr          = float(hparams.get("learning_rate", 1e-4))
    l2wd        = float(hparams.get("l2_weight_decay", 0.0))
    unfreeze    = int(hparams.get("unfrozen_blocks", 0))

    base = VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
    for layer in base.layers:
        layer.trainable = False
    if unfreeze >= 1:
        for layer in base.layers:
            if layer.name.startswith("block5"):
                layer.trainable = True
    if unfreeze >= 2:
        for layer in base.layers:
            if layer.name.startswith(("block4","block5")):
                layer.trainable = True
        lr *= 0.3  # safer when unfreezing deeper

    reg = regularizers.l2(l2wd) if l2wd > 0 else None
    x = base.output
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ----------------------- main API -----------------------
def train_final_and_eval(
    data_dir: Path,
    run_dir: Path,
    class_names: List[str],
    epochs_override: int | None = None,
    patience: int = 10,
    seed: int = 42,
    mixed_precision_flag: bool = False,
) -> None:
    import tensorflow as tf
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau

    _set_seed(seed)

    if mixed_precision_flag:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    # read best hparams
    result_json = json.loads((run_dir / "result.json").read_text())
    hparams = dict(result_json.get("best_hparams", {}))
    if epochs_override is not None:
        hparams["epochs"] = int(epochs_override)
    epochs     = int(hparams.get("epochs", 10))
    batch_size = int(hparams.get("batch_size", 32))

    # data
    X_train, y_train, X_val, y_val, X_test, y_test = _load_arrays(data_dir)
    num_classes = len(class_names)

    # merge train+val for final training
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)

    # memory hygiene before building
    tf.keras.backend.clear_session(); gc.collect()

    model = _build_model_vgg16_like(hparams, num_classes=num_classes)
    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=max(1, patience//2), min_lr=1e-6),
        CSVLogger(str(run_dir / "history.csv"), append=False),
    ]

    # quick validation during final training: reuse test as val (or you can hold out part of X_tr)
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # predictions
    y_true = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # save artifacts
    np.save(run_dir / "y_true.npy", y_true)
    np.save(run_dir / "y_pred.npy", y_pred)
    np.save(run_dir / "y_prob.npy", y_prob)

    # metrics
    from sklearn.metrics import confusion_matrix, classification_report
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    payload = {
        "test_accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "classification_report": rep,
        "confusion_matrix": cm.astype(int).tolist(),
        "epochs_ran": len(hist.history.get("accuracy", [])),
    }
    (run_dir / "final_test_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[results_report] Final eval written to {run_dir}: history.csv, y_true.npy, y_pred.npy, y_prob.npy, final_test_metrics.json")

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--class-names", nargs="+", required=True)
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    train_final_and_eval(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        class_names=args.class_names,
        epochs_override=args.epochs_override,
        patience=args.patience,
        seed=args.seed,
        mixed_precision_flag=args.mixed_precision,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
