#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from results_report import train_final_and_eval

def main():
    ap = argparse.ArgumentParser(description="Train final best model for a single run folder.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="Folder containing result.json with best_hparams")
    ap.add_argument("--class-names", nargs="+", required=True)
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    j = args.run_dir / "result.json"
    if not j.exists():
        raise SystemExit(f"{j} not found (expected)")
    with open(j, "r") as f:
        res = json.load(f)
    best_h = res.get("best_hparams") or res.get("best_hyperparams") or res.get("best")
    if best_h is None:
        raise SystemExit("best_hparams not found in result.json")

    train_final_and_eval(
        hparams=best_h,
        data_dir=args.data_dir,
        out_dir=args.run_dir,
        class_names=args.class_names,
        epochs_override=args.epochs_override,
        patience=args.patience,
        seed=args.seed,
        mixed_precision_flag=args.mixed_precision,
    )

if __name__ == "__main__":
    main()
