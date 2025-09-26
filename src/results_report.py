#!/usr/bin/env python3
import argparse, json, os, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

def set_global_seed(seed: int):
    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def maybe_enable_mixed_precision(enable: bool):
    if enable:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[results_report] Mixed precision enabled (float16 compute).")
        except Exception as e:
            print(f"[results_report] Mixed precision not available: {e}")

def load_arrays(data_dir: Path):
    print(f"[results_report] Loading arrays from: {data_dir}")
    train_X = np.load(data_dir / "train_X.npy")
    train_y = np.load(data_dir / "train_y.npy")
    val_X = np.load(data_dir / "val_X.npy")
    val_y = np.load(data_dir / "val_y.npy")
    test_X = np.load(data_dir / "test_X.npy")
    test_y = np.load(data_dir / "test_y.npy")
    return train_X, train_y, val_X, val_y, test_X, test_y

def build_model(num_classes: int, h: Dict):
    dense_units = int(h["dense_units"])
    dropout = float(h["dropout_rate"])
    lr = float(h["learning_rate"])
    l2wd = float(h.get("l2_weight_decay", 0.0))
    unfreeze = int(h.get("unfrozen_blocks", 0))

    base = VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
    # Freeze all first
    for layer in base.layers:
        layer.trainable = False
    # Unfreeze last 'unfreeze' blocks (each VGG block starts with 'block{n}_')
    if unfreeze > 0:
        blocks = [1,2,3,4,5]
        to_unfreeze = set(blocks[-unfreeze:])
        for layer in base.layers:
            # detect block number
            for b in to_unfreeze:
                if layer.name.startswith(f"block{b}_"):
                    layer.trainable = True
                    break

    reg = regularizers.l2(l2wd) if l2wd > 0 else None

    x = layers.Input(shape=(224,224,3))
    y = base(x, training=False)
    y = layers.Flatten()(y)
    y = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(num_classes, activation="softmax", dtype="float32")(y)  # force float32 for mixed precision stability

    model = models.Model(x, y)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_val(model, h: Dict, train_X, train_y, val_X, val_y, epochs_default: int=20, patience: int=5):
    epochs = int(h.get("epochs", epochs_default))
    bs = int(h["batch_size"])
    ckpt_path = Path(h.get("checkpoint_path", "tmp_best.h5"))
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=max(1,patience//2), min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1),
    ]
    hist = model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=epochs, batch_size=bs, verbose=2, callbacks=callbacks)
    # Reload best
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))
    return hist

def train_final_and_eval(hparams: Dict, data_dir: Path, out_dir: Path, class_names: List[str], epochs_override: int=None, patience: int=8, seed: int=42, mixed_precision_flag: bool=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    maybe_enable_mixed_precision(mixed_precision_flag)

    train_X, train_y, val_X, val_y, test_X, test_y = load_arrays(data_dir)
    num_classes = len(class_names)

    # One-hot if needed
    if train_y.ndim == 1 or train_y.shape[-1] != num_classes:
        from tensorflow.keras.utils import to_categorical
        train_y = to_categorical(train_y, num_classes)
        val_y = to_categorical(val_y, num_classes)
        test_y = to_categorical(test_y, num_classes)

    # Merge train+val
    X_tr = np.concatenate([train_X, val_X], axis=0)
    y_tr = np.concatenate([train_y, val_y], axis=0)

    h = dict(hparams)
    if epochs_override is not None:
        h["epochs"] = int(epochs_override)

    # Train
    model = build_model(num_classes, h)
    h["checkpoint_path"] = str(out_dir / "best_weights.h5")
    _ = train_val(model, h, X_tr, y_tr, test_X, test_y, epochs_default=h.get("epochs", 20), patience=patience)

    # Evaluate on test
    y_prob = model.predict(test_X, batch_size=int(h["batch_size"]), verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(test_y, axis=1)

    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    final = {
        "best_hparams": hparams,
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "best_weights_path": "best_weights.h5",
        "seed": seed,
    }
    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"[results_report] Wrote {out_dir/'final_test_metrics.json'} (acc={acc:.4f})")

    # Confusion matrix plots
    plot_confusion_matrices(cm, class_names, out_dir)

def plot_confusion_matrices(cm: np.ndarray, class_names: List[str], out_dir: Path):
    # Counts
    fig = plt.figure(figsize=(5,4))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (counts)")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i,j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_counts.png", dpi=160)
    plt.close(fig)

    # Row-normalized
    cmn = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    fig = plt.figure(figsize=(5,4))
    im = plt.imshow(cmn, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            plt.text(j, i, f"{cmn[i,j]*100:.1f}%", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_normalized.png", dpi=160)
    plt.close(fig)

def load_result_json(run_dir: Path) -> Dict:
    j = run_dir / "result.json"
    if not j.exists():
        raise FileNotFoundError(f"{j} not found – expected FOA/PSO/GWO result file with best_hparams & history.")
    with open(j, "r") as f:
        return json.load(f)

def plot_convergence(all_histories: Dict[str, List[List[float]]], out_path: Path):
    # all_histories[label] = list of histories (each a list of best val acc per iter)
    plt.figure(figsize=(7,4.5))
    for label, runs in all_histories.items():
        if not runs: 
            continue
        # pad to same length
        maxL = max(len(r) for r in runs)
        aligned = [r + [r[-1]]*(maxL-len(r)) for r in runs]
        arr = np.array(aligned)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(1, maxL+1)
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean-std, mean+std, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Convergence (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Aggregate optimizer runs, train final best models, and create figures.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("report"))
    ap.add_argument("--run", action="append", default=[], help='Add a run like "FOA:/path/to/run_dir" (label:path). Can repeat.')
    ap.add_argument("--class-names", nargs="+", required=True)
    ap.add_argument("--no-auto-final-train", action="store_true")
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    maybe_enable_mixed_precision(args.mixed_precision)

    # Summary holders
    rows = []
    label_to_histories: Dict[str, List[List[float]]] = {}
    label_to_test_accs: Dict[str, List[float]] = {}

    # Process each run
    for spec in args.run:
        if ":" not in spec:
            raise SystemExit(f'Bad --run "{spec}". Use "LABEL:/path"')
        label, path = spec.split(":", 1)
        run_dir = Path(path)
        print(f"[results_report] Processing {label} at {run_dir}")

        try:
            res = load_result_json(run_dir)
        except Exception as e:
            print(f"[results_report] Skipping (no result.json): {e}")
            continue

        best_h = res.get("best_hparams") or res.get("best_hyperparams") or res.get("best")
        history = res.get("history_best_acc") or res.get("history") or []
        label_to_histories.setdefault(label, []).append(list(history))

        final_metrics = run_dir / "final_test_metrics.json"
        if final_metrics.exists():
            with open(final_metrics, "r") as f:
                fm = json.load(f)
            test_acc = float(fm.get("test_accuracy", float("nan")))
        elif not args.no_auto_final_train and best_h is not None:
            # train now
            out_dir = run_dir  # write back into the run folder
            try:
                train_final_and_eval(
                    hparams=best_h,
                    data_dir=args.data_dir,
                    out_dir=out_dir,
                    class_names=args.class_names,
                    epochs_override=args.epochs_override,
                    patience=args.patience,
                    seed=args.seed,
                    mixed_precision_flag=args.mixed_precision,
                )
                with open(run_dir/"final_test_metrics.json","r") as f:
                    fm = json.load(f)
                test_acc = float(fm.get("test_accuracy", float("nan")))
            except Exception as e:
                print(f"[results_report] Failed training final for {label} at {run_dir}: {e}")
                test_acc = float("nan")
        else:
            print(f"[results_report] No final_test_metrics.json and auto-final-train disabled or missing best_hparams.")
            test_acc = float("nan")

        best_val = float(res.get("best_val_acc", res.get("best_accuracy", float("nan"))))
        rows.append({"label": label, "run_path": str(run_dir), "best_val_acc": best_val, "final_test_acc": test_acc})
        label_to_test_accs.setdefault(label, []).append(test_acc)

    # Save tables
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.out / "runs_summary.csv", index=False)
        print(f"[results_report] Wrote {args.out / 'runs_summary.csv'}")

    # Aggregates per label
    agg_rows = []
    for label, accs in label_to_test_accs.items():
        arr = np.array([a for a in accs if not (a!=a)])  # filter NaN
        if arr.size:
            agg_rows.append({
                "label": label,
                "n": int(arr.size),
                "test_acc_mean": float(arr.mean()),
                "test_acc_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            })
    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(args.out / "labels_aggregate.csv", index=False)
        print(f"[results_report] Wrote {args.out / 'labels_aggregate.csv'}")

    # Convergence figure
    if label_to_histories:
        plot_convergence(label_to_histories, args.out / "convergence.png")
        print(f"[results_report] Wrote {args.out / 'convergence.png'}")

    # Readme
    with open(args.out / "README.txt", "w") as f:
        f.write("Files:\n- runs_summary.csv (per run)\n- labels_aggregate.csv (per label mean±std)\n- convergence.png\n- For each run: final_test_metrics.json, best_weights.h5, confusion_*.png\n")
    print(f"[results_report] All done.")

if __name__ == "__main__":
    main()
