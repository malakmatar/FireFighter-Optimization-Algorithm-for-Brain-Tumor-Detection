
"""
pso.py  (FFO-adapted, single file)
----------------------------------
impelemntatin of the particle swarm optimizer
Usage examples:
    python pso.py --mode test --data-dir . --results-dir results --seed 42
    python pso.py --mode full --data-dir . --results-dir results --seed 1 --subset-frac 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

#reuse same functions as ffo to be consisitent
from ffo import SpaceSpec, ParamSpace, load_arrays


# --------------------------- Core PSO (single-file) --------------------------

Array = np.ndarray

@dataclass
class PSOResult:
    best_position: Array
    best_score: float
    history: List[Dict[str, Any]]
    evaluations: int

class PSO:
    """
    Velocity update:
        v <- v + 2*rand()*(pbest - x) + 2*rand()*(gbest - x)
        x <- x + v

    No inertia/constriction; global-best topology. Positions are clipped to bounds.
    Velocities can be softly clipped relative to range to avoid explosions.
    """

    def __init__(
        self,
        objective,
        bounds: Array,
        n_particles: int = 20,
        n_iters: int = 50,
        seed: Optional[int] = None,
        maximize: bool = False,
        v_init_scale: float = 0.1,
        v_clip: Optional[float] = 0.5,
        c1: float = 2.0,
        c2: float = 2.0,
    ) -> None:
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        assert self.bounds.ndim == 2 and self.bounds.shape[1] == 2, "bounds must be (D, 2)"
        self.D = self.bounds.shape[0]
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.maximize = bool(maximize)
        self.v_init_scale = float(v_init_scale)
        self.v_clip = v_clip
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.rng = np.random.default_rng(seed)

    def _eval(self, x: Array) -> float:
        val = self.objective(x)
        return -val if self.maximize else val

    def run(self, x0: Optional[Array] = None) -> PSOResult:
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        span = high - low

        if x0 is None:
            X = self.rng.uniform(low, high, size=(self.n_particles, self.D))
        else:
            X = np.asarray(x0, dtype=float)
            assert X.shape == (self.n_particles, self.D)
            X = np.clip(X, low, high)

        # initial v
        V = self.rng.uniform(-1.0, 1.0, size=(self.n_particles, self.D)) * (self.v_init_scale * span)

        # Evaluate initials
        scores = np.apply_along_axis(self._eval, 1, X)
        pbest_pos = X.copy()
        pbest_scores = scores.copy()
        g_idx = np.argmin(pbest_scores)
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_score = pbest_scores[g_idx]

        history: List[Dict[str, Any]] = [{
            "iter": 0,
            "gbest_score": -gbest_score if self.maximize else gbest_score,
            "gbest_position": gbest_pos.copy(),
        }]
        evaluations = int(self.n_particles)

        for it in range(1, self.n_iters + 1):
            r1 = self.rng.random(size=(self.n_particles, self.D))
            r2 = self.rng.random(size=(self.n_particles, self.D))
            V += self.c1 * r1 * (pbest_pos - X) + self.c2 * r2 * (gbest_pos - X)

            if self.v_clip is not None and self.v_clip > 0:
                vmax = self.v_clip * span
                V = np.clip(V, -vmax, vmax)

            X = X + V

            # for bounds
            out_low = X < low
            out_high = X > high
            if np.any(out_low | out_high):
                X = np.clip(X, low, high)
                V[out_low] *= -0.5
                V[out_high] *= -0.5

            scores = np.apply_along_axis(self._eval, 1, X)
            evaluations += self.n_particles

            improved = scores < pbest_scores
            pbest_pos[improved] = X[improved]
            pbest_scores[improved] = scores[improved]

            g_idx = np.argmin(pbest_scores)
            if pbest_scores[g_idx] < gbest_score:
                gbest_score = pbest_scores[g_idx]
                gbest_pos = pbest_pos[g_idx].copy()

            history.append({
                "iter": it,
                "gbest_score": -gbest_score if self.maximize else gbest_score,
                "gbest_position": gbest_pos.copy(),
            })

        final_score = -gbest_score if self.maximize else gbest_score
        return PSOResult(best_position=gbest_pos, best_score=final_score, history=history, evaluations=evaluations)


# --------------------------- Objective & caching -----------------------------

def _eval_accuracy(hparams: Dict[str, float | int], arrays) -> float:
    from train_wrapper import evaluate_model  # project-local import
    train_X, train_y, val_X, val_y = arrays
    print(f"[PSO] evaluate_model() starting with hparams={hparams}", flush=True)
    acc = float(evaluate_model(hparams, train_X, train_y, val_X, val_y))
    print(f"[PSO] evaluate_model() done -> val_acc={acc:.4f}", flush=True)
    return acc


def minimize_neg_accuracy(space: ParamSpace, arrays, cache_file: Path | None, fixed_epochs: int):
    """Return f(v) that maps normalized vector v in [0,1]^D to -val_acc with tiny JSON cache."""
    cache: Dict[str, float] = {}
    if cache_file and cache_file.exists():
        try:
            cache.update(json.loads(cache_file.read_text()))
        except Exception:
            pass

    def _hp_key(h: Dict[str, float | int]) -> str:
        return json.dumps({k: h[k] for k in sorted(h.keys())}, sort_keys=True)

    def _obj(v: np.ndarray) -> float:
        h = space.vec_to_hparams(v, fixed_epochs=fixed_epochs)
        k = _hp_key(h)
        if k in cache:
            return -float(cache[k])
        acc = _eval_accuracy(h, arrays)
        if cache_file:
            cache[k] = float(acc)
            try:
                cache_file.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass
        return -float(acc)

    return _obj


# --------------------------- Persistence ------------------------------------

def save_run(out_dir: Path, spec: "SpaceSpec", cfg: "PSOConfig",
             best_v: np.ndarray, space: ParamSpace, best_score: float,
             history_acc: List[float], wall_time: float, cache_file: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "space.json").write_text(json.dumps(asdict(spec), indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    best_h = space.vec_to_hparams(best_v, fixed_epochs=cfg.fixed_epochs)
    (out_dir / "result.json").write_text(json.dumps({
        "best_hparams": best_h,
        "best_val_accuracy": float(-best_score),  # score is -acc
        "history_best_acc": [float(a) for a in history_acc],
        "eval_count": int(cfg.n_particles * min(cfg.n_iters, len(history_acc))),
        "wall_time_sec": float(wall_time),
    }, indent=2))

    # NEW: write convergence CSV here (out_dir is known)
    try:
        import csv
        with open(out_dir / "search_history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "best_val_acc"])
            for i, acc in enumerate(history_acc):
                w.writerow([i, float(acc)])
    except Exception as e:
        print(f"[PSO] WARN: failed to write search_history.csv: {e}", flush=True)

    if cache_file and cache_file.exists():
        try:
            (out_dir / "objective_cache.json").write_text(cache_file.read_text())
        except Exception:
            pass



# --------------------------- Config & CLI -----------------------------------

@dataclass
class PSOConfig:
    n_particles: int
    n_iters: int
    c1: float
    c2: float
    seed: int
    fixed_epochs: int

def make_cfg(mode: str, seed: int, fixed_epochs: int) -> PSOConfig:
    if mode == "test":
        return PSOConfig(n_particles=4, n_iters=3, c1=2.0, c2=2.0, seed=seed, fixed_epochs=fixed_epochs)
    elif mode == "full":
        return PSOConfig(n_particles=24, n_iters=20, c1=2.0, c2=2.0, seed=seed, fixed_epochs=fixed_epochs)
    else:
        raise ValueError("mode must be 'test' or 'full'")

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PSO (1995 simplified) for CNN hparam tuning (ffo-compatible CLI, single-file)")
    ap.add_argument("--mode", choices=["test", "full"], default="test")
    ap.add_argument("--data-dir", type=str, default=".")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=str, default=".pso_cache.json")
    ap.add_argument("--subset-frac", type=float, default=None,
                    help="fraction of TRAIN set to use with stratified sampling (e.g., 0.25). If omitted: test mode defaults to 0.25; full mode uses 1.0")
    # fixed epochs per mode (match ffo.py flags)
    ap.add_argument("--epochs-fixed-test", type=int, default=3, help="fixed epochs used in test mode")
    ap.add_argument("--epochs-fixed-full", type=int, default=10, help="fixed epochs used in full mode")
    # search space overrides (identical to ffo.py)
    ap.add_argument("--dense-min", type=int, default=128)
    ap.add_argument("--dense-max", type=int, default=512)
    ap.add_argument("--dropout-min", type=float, default=0.25)
    ap.add_argument("--dropout-max", type=float, default=0.55)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--lr-max", type=float, default=5e-4)
    ap.add_argument("--batch-min", type=int, default=16)
    ap.add_argument("--batch-max", type=int, default=32)
    ap.add_argument("--l2-min", type=float, default=1e-6)
    ap.add_argument("--l2-max", type=float, default=1e-3)
    ap.add_argument("--particles", type=int, default=None, help="override n_particles")
    ap.add_argument("--iters", type=int, default=None, help="override n_iters")
    args = ap.parse_args(argv)


    rng = np.random.default_rng(args.seed)

    spec = SpaceSpec(
        dense_units_min=args.dense_min, dense_units_max=args.dense_max,
        dropout_min=args.dropout_min, dropout_max=args.dropout_max,
        lr_min=args.lr_min, lr_max=args.lr_max,
        batch_min=args.batch_min, batch_max=args.batch_max,
        l2_min=args.l2_min, l2_max=args.l2_max,
    )
    space = ParamSpace(spec)

    arrays = load_arrays(Path(args.data_dir))

    # subset logic identical to ffo.py
    subset_frac = args.subset_frac
    if subset_frac is None:
        subset_frac = 0.25 if args.mode == "test" else None
    if subset_frac is not None and 0 < subset_frac < 1:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            train_X, train_y, val_X, val_y = arrays
            sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_frac, random_state=args.seed)
            (idx_small, _), = sss.split(train_X, train_y)
            train_X = train_X[idx_small]
            train_y = train_y[idx_small]
            arrays = (train_X, train_y, val_X, val_y)
            print(f"[PSO] Using subset of train: {len(idx_small)} samples (~{subset_frac*100:.0f}%)", flush=True)
        except Exception as e:
            print(f"[PSO] WARN: subset sampling failed ({e}); falling back to full train set.", flush=True)

    fixed_epochs = int(args.epochs_fixed_test if args.mode == "test" else args.epochs_fixed_full)
    print(f"[PSO] Fixed epochs this run: {fixed_epochs}", flush=True)

    cfg = make_cfg(args.mode, seed=args.seed, fixed_epochs=fixed_epochs)
    objective = minimize_neg_accuracy(space, arrays, Path(args.cache) if args.cache else None,
                                      fixed_epochs=fixed_epochs)

    # overrides
    if args.particles is not None:
        cfg.n_particles = int(args.particles)
    if args.iters is not None:
        cfg.n_iters = int(args.iters)

    # PSO setup on normalized hypercube [0,1]^D
    D = space.dim
    bounds = np.array([[0.0, 1.0]] * D, dtype=float)

    # Run
    t0 = time.time()
    opt = PSO(
        objective=objective,
        bounds=bounds,
        n_particles=cfg.n_particles,
        n_iters=cfg.n_iters,
        seed=cfg.seed,
        maximize=False,   # minimizing -val_acc
        c1=cfg.c1, c2=cfg.c2,
        v_init_scale=0.1,
        v_clip=0.5,
    )
    result = opt.run()
    wall = time.time() - t0

    # Convert to reporting
    history_acc = [float(-h["gbest_score"]) for h in result.history]
    best_v = result.best_position.copy()
    best_acc = float(-result.best_score)
    traj_df = pd.DataFrame({
        "iteration": list(range(len(history_acc))),  # 0 .. n_iters
        "best_val_acc": history_acc
    })

    # Reveal best hyperparams
    best_h = space.vec_to_hparams(best_v, fixed_epochs=fixed_epochs)
    print("\\n[PSO] Best validation accuracy:", f"{best_acc:.4f}", flush=True)
    print("[PSO] Best hyperparameters:", flush=True)
    for k, v in best_h.items():
        print(f"  - {k}: {v}", flush=True)

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / f"pso_{args.mode}_{stamp}"
    save_run(run_dir, spec, cfg, best_v, space, result.best_score, history_acc, wall, Path(args.cache) if args.cache else None)
    print(f"\\n[PSO] Results saved to: {run_dir}\\n", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
