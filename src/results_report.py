#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
results_report.py — compact, robust reporter with better convergence handling

What it does
------------
• Scans one or more run directories and collects:
  - result.json (best hparams, best val acc, eval_count, wall_time_sec)
  - config.json / space.json (optional metadata)
  - final_test_metrics.json, y_true.npy, y_pred.npy, y_prob.npy (if present)
  - search_history.csv (preferred for convergence)
  - history_best_acc from result.json (fallback for convergence)
  - history.csv from final training (last fallback for convergence)

• Optionally auto-runs `train_best_from_result.py` to produce test predictions
  unless `--no-auto-final-train` is set (requires --data-dir).

• Writes CSVs:
  - summary_runs.csv
  - summary_by_algorithm.csv
  - best_hparams.csv
  - summary.csv (one-line rollup)

• Produces a single PDF (thesis_report.pdf) with:
  - Convergence plot (by algorithm)
  - (If predictions present) Confusion matrix, per-class precision/recall/F1,
    ROC/PR curves (one panel with subplots), and a simple calibration curve.

CLI examples
------------
# Single run (visuals only)
python src/results_report.py \
  --results-root runs/pso_smoke/pso_test_20250928_190025 \
  --out reports/PSO_smoke \
  --class-names glioma_tumor meningioma_tumor pituitary_tumor no_tumor \
  --no-auto-final-train \
  --algo-name PSO

# Aggregate all runs under a folder (auto-final-train if needed and data-dir provided)
python src/results_report.py \
  --results-root runs/ \
  --data-dir . \
  --out reports/combined \
  --class-names glioma_tumor meningioma_tumor pituitary_tumor no_tumor
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# --------------------------- I/O helpers -------------------------------------


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _np_load(p: Path) -> Optional[np.ndarray]:
    try:
        return np.load(p, allow_pickle=False)
    except Exception:
        return None


def _is_run_dir(d: Path) -> bool:
    # A run dir is recognized by the presence of result.json
    return d.is_dir() and (d / "result.json").exists()


def _find_run_dirs(root: Path) -> List[Path]:
    if _is_run_dir(root):
        return [root]
    runs: List[Path] = []
    for p in root.rglob("*"):
        if _is_run_dir(p):
            runs.append(p)
    return sorted(runs)


# --------------------------- Models / dataclasses ----------------------------


@dataclass
class RunArtifacts:
    run_dir: Path
    algorithm: str
    result: Dict
    config: Optional[Dict]
    space: Optional[Dict]
    search_history: Optional[pd.DataFrame]  # columns: iteration, best_val_acc
    train_history: Optional[pd.DataFrame]   # final training history.csv (epochs)
    y_true: Optional[np.ndarray]
    y_pred: Optional[np.ndarray]
    y_prob: Optional[np.ndarray]
    final_metrics: Optional[Dict]


# --------------------------- Collection --------------------------------------


def _infer_algo_name(run_dir: Path, default_algo: Optional[str]) -> str:
    # Prefer explicit CLI --algo-name; else infer from directory name
    if default_algo:
        return default_algo
    name = run_dir.name.lower()
    if "pso" in name:
        return "PSO"
    if "ffo" in name or "foa" in name:
        return "FFO"
    if "gwo" in name:
        return "GWO"
    return "UNKNOWN"


def _load_convergence(run_dir: Path, result_json: Dict) -> Optional[pd.DataFrame]:
    """
    Preferred order:
    1) search_history.csv -> columns: iteration, best_val_acc
    2) history_best_acc in result.json -> synthesize DataFrame
    3) history.csv (final train) -> fallback plotted as epoch vs val_accuracy
    """
    # 1) CSV
    csv_df = _read_csv(run_dir / "search_history.csv")
    if isinstance(csv_df, pd.DataFrame) and not csv_df.empty and \
       {"iteration", "best_val_acc"}.issubset(csv_df.columns):
        return csv_df

    # 2) JSON fallback
    seq = result_json.get("history_best_acc")
    if isinstance(seq, list) and len(seq) > 0:
        return pd.DataFrame({
            "iteration": np.arange(len(seq), dtype=int),
            "best_val_acc": np.array(seq, dtype=float),
        })

    # 3) Final training history
    h = _read_csv(run_dir / "history.csv")
    if isinstance(h, pd.DataFrame) and not h.empty:
        # normalize to the same column names so plotter can still show something
        cols = [c for c in h.columns if "val_acc" in c]
        if cols:
            return pd.DataFrame({
                "iteration": np.arange(len(h), dtype=int),
                "best_val_acc": h[cols[0]].astype(float).values,
            })
    return None


def _collect_run(run_dir: Path, algo_name_arg: Optional[str]) -> RunArtifacts:
    result = _read_json(run_dir / "result.json") or {}
    config = _read_json(run_dir / "config.json")
    space = _read_json(run_dir / "space.json")
    algo = _infer_algo_name(run_dir, algo_name_arg)

    search_hist = _load_convergence(run_dir, result)
    train_hist = _read_csv(run_dir / "history.csv")

    y_true = _np_load(run_dir / "y_true.npy")
    y_pred = _np_load(run_dir / "y_pred.npy")
    y_prob = _np_load(run_dir / "y_prob.npy")
    final_metrics = _read_json(run_dir / "final_test_metrics.json")

    return RunArtifacts(
        run_dir=run_dir,
        algorithm=algo,
        result=result,
        config=config,
        space=space,
        search_history=search_hist,
        train_history=train_hist,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        final_metrics=final_metrics,
    )


def _has_predictions(art: RunArtifacts) -> bool:
    return (
        art.y_true is not None and
        art.y_pred is not None and
        art.y_prob is not None and
        art.final_metrics is not None
    )


def _auto_final_train(
    art: RunArtifacts,
    data_dir: Optional[Path],
    class_names: List[str],
    patience: int = 10,
    seed: int = 42,
    epochs_override: Optional[int] = None,
) -> None:
    """
    Calls train_best_from_result.py to generate predictions in-place.
    """
    if data_dir is None:
        return
    py = Path(os.environ.get("VIRTUAL_ENV", "")) / "bin" / "python"
    py = py if py.exists() else Path("python")

    cmd = [
        str(py),
        str(Path(__file__).parent / "train_best_from_result.py"),
        "--data-dir", str(data_dir),
        "--run-dir", str(art.run_dir),
        "--patience", str(patience),
        "--seed", str(seed),
        "--class-names", *class_names,
    ]
    if epochs_override is not None:
        cmd.extend(["--epochs-override", str(epochs_override)])

    print(f"[results_report] Auto-running final training to produce predictions: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

    # Refresh artifacts after training
    art.y_true = _np_load(art.run_dir / "y_true.npy")
    art.y_pred = _np_load(art.run_dir / "y_pred.npy")
    art.y_prob = _np_load(art.run_dir / "y_prob.npy")
    art.final_metrics = _read_json(art.run_dir / "final_test_metrics.json")


# --------------------------- Plot helpers ------------------------------------


def _plot_convergence(ax, grouped_hist: Dict[str, List[pd.DataFrame]]) -> None:
    """
    grouped_hist: algo -> list of DataFrames with columns (iteration, best_val_acc)
    """
    for algo, dfs in grouped_hist.items():
        # align by iteration index; compute mean±std
        max_len = max((len(df) for df in dfs if df is not None and not df.empty), default=0)
        if max_len == 0:
            continue
        mat = []
        for df in dfs:
            if df is None or df.empty:
                continue
            arr = df["best_val_acc"].to_numpy()
            if len(arr) < max_len:
                pad = np.full(max_len - len(arr), arr[-1], dtype=float)
                arr = np.concatenate([arr, pad], axis=0)
            mat.append(arr)
        if not mat:
            continue
        M = np.vstack(mat)
        mean = M.mean(axis=0)
        std = M.std(axis=0)
        xs = np.arange(max_len)
        ax.plot(xs, mean, label=algo)
        if max_len > 1 and (std > 0).any():
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
        elif max_len == 1:
            ax.scatter(xs, mean, label=f"{algo} (single pt)")
    ax.set_title("Search Convergence (Best Validation Accuracy)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Val Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()


def _plot_confusion(ax, y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Normalized Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cmn):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_roc_pr(fig, y_true, y_prob, class_names):
    K = len(class_names)
    y_true_oh = np.eye(K, dtype=int)[y_true.astype(int)]
    # ROC
    ax1 = fig.add_subplot(1, 2, 1)
    for k in range(K):
        fpr, tpr, _ = roc_curve(y_true_oh[:, k], y_prob[:, k])
        ax1.plot(fpr, tpr, label=f"{class_names[k]} (AUC={auc(fpr, tpr):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("ROC Curves")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    # PR
    ax2 = fig.add_subplot(1, 2, 2)
    for k in range(K):
        p, r, _ = precision_recall_curve(y_true_oh[:, k], y_prob[:, k])
        ap = average_precision_score(y_true_oh[:, k], y_prob[:, k])
        ax2.plot(r, p, label=f"{class_names[k]} (AP={ap:.3f})")
    ax2.set_title("Precision-Recall Curves")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)


def _plot_calibration(ax, y_true, y_prob, n_bins: int = 10):
    """
    Simple reliability diagram: bin max prob vs empirical accuracy.
    """
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true.astype(int)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins) - 1
    bin_acc, bin_conf, bin_cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bin_acc.append(correct[m].mean())
            bin_conf.append(conf[m].mean())
            bin_cnt.append(m.sum())
        else:
            bin_acc.append(np.nan)
            bin_conf.append((bins[b] + bins[b + 1]) / 2)
            bin_cnt.append(0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.scatter(bin_conf, bin_acc, s=np.clip(np.array(bin_cnt) * 2, 10, 200))
    ax.set_title("Calibration (Reliability Diagram)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)


# --------------------------- Main reporting ----------------------------------


def build_and_save_report(
    runs: List[RunArtifacts],
    out_dir: Path,
    class_names: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSVs: per-run summary ----
    rows = []
    for art in runs:
        rj = art.result or {}
        fm = art.final_metrics or {}
        rows.append({
            "run_dir": str(art.run_dir),
            "algorithm": art.algorithm,
            "best_val_accuracy": rj.get("best_val_accuracy"),
            "eval_count": rj.get("eval_count"),
            "wall_time_sec": rj.get("wall_time_sec"),
            "test_accuracy": fm.get("test_accuracy"),
            "test_macro_f1": fm.get("macro_f1"),
            "test_macro_precision": fm.get("macro_precision"),
            "test_macro_recall": fm.get("macro_recall"),
        })
    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(out_dir / "summary_runs.csv", index=False)
    print(f"[results_report] Wrote {out_dir/'summary_runs.csv'}")

    # ---- CSV: best_hparams ----
    bh_rows = []
    for art in runs:
        rj = art.result or {}
        hp = rj.get("best_hparams", {})
        row = {"run_dir": str(art.run_dir), "algorithm": art.algorithm}
        row.update(hp)
        bh_rows.append(row)
    pd.DataFrame(bh_rows).to_csv(out_dir / "best_hparams.csv", index=False)
    print(f"[results_report] Wrote {out_dir/'best_hparams.csv'}")

    # ---- CSV: summary_by_algorithm ----
    if not df_runs.empty:
        df_alg = (
            df_runs.groupby("algorithm")[["best_val_accuracy", "test_accuracy"]]
            .agg(["mean", "std", "count"])
        )
        df_alg.to_csv(out_dir / "summary_by_algorithm.csv")
        print(f"[results_report] Wrote {out_dir/'summary_by_algorithm.csv'}")

    # ---- CSV: one-line summary ----
    if not df_runs.empty:
        overall = {
            "n_runs": len(df_runs),
            "mean_best_val_acc": float(pd.to_numeric(df_runs["best_val_accuracy"], errors="coerce").mean()),
            "mean_test_acc": float(pd.to_numeric(df_runs["test_accuracy"], errors="coerce").mean()),
        }
        pd.DataFrame([overall]).to_csv(out_dir / "summary.csv", index=False)
        print(f"[results_report] Wrote {out_dir/'summary.csv'}")

    # ---- PDF report ----
    pdf_path = out_dir / "thesis_report.pdf"
    with PdfPages(pdf_path) as pdf:

        # Convergence panel (by algorithm)
        per_algo_hist: Dict[str, List[pd.DataFrame]] = {}
        for art in runs:
            if art.search_history is not None and not art.search_history.empty:
                per_algo_hist.setdefault(art.algorithm, []).append(art.search_history)
            else:
                per_algo_hist.setdefault(art.algorithm, []).append(pd.DataFrame())

        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_convergence(ax, per_algo_hist)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # If at least one run has predictions, produce evaluation figures for each run
        for art in runs:
            if not _has_predictions(art):
                continue

            # Confusion
            fig, ax = plt.subplots(figsize=(6, 5))
            _plot_confusion(ax, art.y_true, art.y_pred, class_names)
            ax.set_title(f"Confusion Matrix — {art.algorithm}\n{art.run_dir.name}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ROC + PR
            fig = plt.figure(figsize=(10, 4))
            _plot_roc_pr(fig, art.y_true, art.y_prob, class_names)
            fig.suptitle(f"ROC & PR — {art.algorithm}\n{art.run_dir.name}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Calibration
            fig, ax = plt.subplots(figsize=(6, 4))
            _plot_calibration(ax, art.y_true, art.y_prob, n_bins=10)
            ax.set_title(f"Calibration — {art.algorithm}\n{art.run_dir.name}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[results_report] Wrote {pdf_path}")


# --------------------------- CLI ---------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("results_report")
    ap.add_argument("--results-root", type=str, required=True,
                    help="Run directory OR a root containing many run subfolders")
    ap.add_argument("--out", type=str, required=True, help="Output directory for reports")
    ap.add_argument("--class-names", nargs="+", required=True, help="Class names in label order")
    ap.add_argument("--algo-name", type=str, default=None, help="Override algorithm name for all runs")
    ap.add_argument("--no-auto-final-train", action="store_true",
                    help="Do NOT auto-run final training if predictions are missing")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Folder with preprocessed arrays (.npy). Required if auto-final-train is desired.")
    ap.add_argument("--patience", type=int, default=10, help="Patience used if auto-final-train kicks in")
    ap.add_argument("--seed", type=int, default=42, help="Seed used if auto-final-train kicks in")
    ap.add_argument("--epochs-override", type=int, default=None,
                    help="Force epoch count in auto-final-train (e.g., 0 to only evaluate, 1 for very quick run)")
    args = ap.parse_args(argv)

    root = Path(args.results_root).resolve()
    out_dir = Path(args.out).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    class_names = args.class_names

    run_dirs = _find_run_dirs(root)
    print(f"[results_report] Found {len(run_dirs)} run(s).")
    for r in run_dirs:
        print(f"[results_report]  - {r}")

    arts: List[RunArtifacts] = []
    for rdir in run_dirs:
        art = _collect_run(rdir, args.algo_name)

        # If missing predictions and auto mode allowed, try to produce them
        if not _has_predictions(art):
            if args.no_auto_final_train:
                print(f"[results_report] Missing predictions in {rdir}. Skipping (no auto-final-train).")
            else:
                if data_dir is None:
                    print(f"[results_report] Missing predictions in {rdir} and --data-dir not provided. Skipping.")
                else:
                    _auto_final_train(
                        art,
                        data_dir=data_dir,
                        class_names=class_names,
                        patience=args.patience,
                        seed=args.seed,
                        epochs_override=args.epochs_override,
                    )
        arts.append(art)

    build_and_save_report(arts, out_dir=out_dir, class_names=class_names)
    print("[results_report] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
