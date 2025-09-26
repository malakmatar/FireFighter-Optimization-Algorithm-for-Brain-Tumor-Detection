"""
Firefighter Optimization (FFO) for VGG16 Hyperparameter Tuning

This script performs the following tasks:
1. Defines the search space and maps normalized vectors to concrete hyperparameters
2. Runs FFO to minimize -(validation accuracy) using the evaluate_model objective in the train_wrapper script
3. Handles seeding, logging, checkpointing of best solutions and run metadata
4. CLI supports test/full modes and result directory organization to be able to check at diffrient stages and with
diffrient parameters
5. The CLI also provides the chance to change the limits of the search space and control number of epochs, dense,
dropout, etc. as shown in the example run

Usage:
    python ffo.py --mode test --results-dir results/ffo_runs

example run:
    python -u ffo.py --mode full \
  --epochs-fixed-full 60 \
  --dense-min 128 --dense-max 1024 \
  --dropout-min 0.10 --dropout-max 0.60 \
  --lr-min 3e-6 --lr-max 1e-3 \
  --batch-min 16 --batch-max 96 \
  --l2-min 1e-7 --l2-max 1e-2 \
  --seed 42
"""


from __future__ import annotations
import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# --------------------------- Utilities & seeding ---------------------------------------

def set_global_seed(
    #Set NumPy/Python/TensorFlow seeds for full reproducibility.
seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
        try:
            for d in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            pass
    except Exception:
        pass


# --------------------------- Search space mapping -------------------------------------
@dataclass
class SpaceSpec:
    #Declarative bounds for each hyperparameter (including log-scaled ones).

    dense_units_min: int = 128
    dense_units_max: int = 512
    dropout_min: float = 0.25
    dropout_max: float = 0.55
    lr_min: float = 1e-5
    lr_max: float = 5e-4
    batch_min: int = 16
    batch_max: int = 32
    l2_min: float = 1e-6
    l2_max: float = 1e-3
    unfreeze_max: int = 2  # {0=freeze all, 1=unfreeze block5, 2=unfreeze blocks4–5}

    def clamp(self, name: str, val: float) -> float:
        lo, hi = {
            "dense_units": (self.dense_units_min, self.dense_units_max),
            "dropout_rate": (self.dropout_min, self.dropout_max),
            "learning_rate": (self.lr_min, self.lr_max),
            "batch_size": (self.batch_min, self.batch_max),
            "l2_weight_decay": (self.l2_min, self.l2_max),
        }[name]
        return float(min(max(val, lo), hi))


def _round_multiple(
    #Round integers to nearest multiple (used for batch sizes, etc.).
x: int, m: int) -> int:
    return int(max(m, int(round(x / m) * m)))


class ParamSpace:
    #Bi-directional map between normalized [0,1]^D vectors and real hparams.
    #Order: [dense_units, dropout_rate, log10(lr), batch_size, log10(l2), unfreeze_blocks].
    def __init__(self, spec: SpaceSpec):
        self.s = spec
        self._log_lr_min = math.log10(self.s.lr_min)
        self._log_lr_max = math.log10(self.s.lr_max)
        self._log_l2_min = math.log10(self.s.l2_min)
        self._log_l2_max = math.log10(self.s.l2_max)

    @property
    def dim(self) -> int:
        return 6

    def clip01(self, v: np.ndarray) -> np.ndarray:
        return np.clip(v, 0.0, 1.0)

    def sample_vecs(self, n: int) -> np.ndarray:
        return np.random.rand(n, self.dim)

# --- Mapping from normalized [0,1]^D vector to concrete hyperparameters ---
# We use linear mapping for discrete ranges and log-space for rates.
# This keeps the search numerically stable while exploring meaningful scales.

    def vec_to_hparams(self, v: np.ndarray, fixed_epochs: int) -> Dict[str, float | int]:
        v = self.clip01(np.asarray(v, dtype=float))
        # dense_units (multiple of 16)
        du = int(round(self.s.dense_units_min + v[0] * (self.s.dense_units_max - self.s.dense_units_min)))
        du = _round_multiple(du, 16)
        du = int(self.s.clamp("dense_units", du))
        # dropout
        dr = self.s.dropout_min + v[1] * (self.s.dropout_max - self.s.dropout_min)
        # lr on log scale
        log_lr = self._log_lr_min + v[2] * (self._log_lr_max - self._log_lr_min)
        lr = 10 ** log_lr
        # batch size (multiple of 8)
        bs = int(round(self.s.batch_min + v[3] * (self.s.batch_max - self.s.batch_min)))
        bs = _round_multiple(bs, 8)
        bs = int(self.s.clamp("batch_size", bs))
        # L2 on log scale
        log_l2 = self._log_l2_min + v[4] * (self._log_l2_max - self._log_l2_min)
        l2 = 10 ** log_l2
        l2 = float(self.s.clamp("l2_weight_decay", l2))
        # unfreeze blocks categorical {0,1,2}
        unfreeze = int(round(v[5] * self.s.unfreeze_max))
        unfreeze = int(max(0, min(self.s.unfreeze_max, unfreeze)))
        return {
            "dense_units": du,
            "dropout_rate": float(dr),
            "learning_rate": float(lr),
            "batch_size": bs,
            "l2_weight_decay": l2,
            "unfrozen_blocks": unfreeze,
            # epochs is now FIXED per mode/flag for fairness
            "epochs": int(fixed_epochs),
        }

# --------------------------- Dataset & objective wrapper -------------------------------

def load_arrays(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"[FFO] Loading arrays from: {data_dir}", flush=True)
    train_X = np.load(data_dir / "train_X.npy")
    train_y = np.load(data_dir / "train_y.npy")
    val_X = np.load(data_dir / "val_X.npy")
    val_y = np.load(data_dir / "val_y.npy")
    try:
        print(f"[FFO] Shapes -> train_X:{train_X.shape} train_y:{train_y.shape} val_X:{val_X.shape} val_y:{val_y.shape}", flush=True)
    except Exception:
        pass
    return train_X, train_y, val_X, val_y


def _eval_accuracy(hparams: Dict[str, float | int], arrays) -> float:
    from train_wrapper import evaluate_model  # project-local import
    train_X, train_y, val_X, val_y = arrays
    print(f"[FFO] evaluate_model() starting with hparams={hparams}", flush=True)
    acc = float(evaluate_model(hparams, train_X, train_y, val_X, val_y))
    print(f"[FFO] evaluate_model() done -> val_acc={acc:.4f}", flush=True)
    return acc


# --------------------------- FFO core ---------------------------------
@dataclass
class FFOConfig:
    #All algorithm hyperparameters controlling population size and move operators

    num_agents: int
    max_iter: int
    no_improve_limit: int
    crossover_probability: float
    mutation_probability: float
    step_size: float           # stddev for gaussian moves in normalized space
    initial_temp: float
    cooling_rate: float        # multiply T each iteration (0<cr<1)
    stagnation_perturb_after: int  # iterations with no improvement before applying perturbation
    perturb_base: float        # base intensity for perturbation
    perturb_growth: float      # growth per extra no-improve step beyond threshold
    seed: int


@dataclass
class FFOResult:
    """Container for the best solution, fitness history, and bookkeeping counters."""

    best_hparams: Dict[str, float | int]
    best_val_accuracy: float
    history_best_acc: List[float]
    eval_count: int
    wall_time_sec: float


class FirefighterOptimization:
    #Paper-compliant Firefighter Optimization on a normalized search cube.
    #Minimization internally (we minimize -val_acc)

    def __init__(self, objective_minimize, dim: int, cfg: FFOConfig,
                 bounds=(0.0, 1.0), seed: int = 42, verbose: bool = True):
        self.obj = objective_minimize
        self.dim = dim
        self.lb, self.ub = float(bounds[0]), float(bounds[1])
        self.cfg = cfg
        self.verbose = verbose
        set_global_seed(seed)

        self.agents = np.random.uniform(self.lb, self.ub, size=(cfg.num_agents, dim))
        self.mutation_rates = np.full(cfg.num_agents, 0.1, dtype=float)

        # evaluate initial population
        fits = np.array([self.obj(a) for a in self.agents])
        bi = int(np.argmin(fits))
        self.best_agent = self.agents[bi].copy()
        self.best_fit = float(fits[bi])
        self.no_improve = 0
        self.iteration = 1
        self.eval_count = len(fits)
        self.fitness_history = [float(-self.best_fit)]  # store as ACCURACY (positive)

    def fitness(self, vec: np.ndarray) -> float:
        return self.obj(vec)

    # ----------------------- operators -------------------------------------------------
    def evaluate_agents(self) -> np.ndarray:
        print(f"[FFO] Evaluating population ({len(self.agents)} agents) ...", flush=True)
        fits_list: List[float] = []
        total = len(self.agents)
        for i, a in enumerate(self.agents, start=1):
            f = self.fitness(a)
            fits_list.append(f)
            print(f"[FFO] agent {i}/{total} -> acc={-f:.4f}", flush=True)
        fits = np.asarray(fits_list)
        bi = int(np.argmin(fits))
        if fits[bi] < self.best_fit - 1e-12:
            self.best_fit = float(fits[bi])
            self.best_agent = self.agents[bi].copy()
            self.no_improve = 0
        else:
            self.no_improve += 1
        self.eval_count += len(fits)
        return fits

    def crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.random.randint(1, self.dim)
        c1 = np.concatenate([a[:p], b[p:]])
        c2 = np.concatenate([b[:p], a[p:]])
        return c1, c2

    def local_search(self, agent: np.ndarray, temp: float, idx: int) -> np.ndarray:
        best_local = agent.copy()
        best_fit = self.obj(best_local)
        self.eval_count += 1
        tries = 6 + (self.no_improve // 50)
        sigma = self.cfg.step_size * self.mutation_rates[idx]
        for _ in range(tries):
            cand = best_local + np.random.normal(0.0, sigma, size=self.dim)
            cand = np.clip(cand, self.lb, self.ub)
            f = self.obj(cand)
            self.eval_count += 1
            if f < best_fit or np.random.rand() < math.exp((best_fit - f) / max(1e-12, temp)):
                best_local, best_fit = cand, f
        return best_local

    def apply_perturbation(self, agent: np.ndarray, intensity: float) -> np.ndarray:
        direction = self.best_agent - agent
        noise = np.random.normal(0.0, intensity, size=self.dim)
        cand = agent + noise * direction
        return np.clip(cand, self.lb, self.ub)

    def cooling_temperature(self) -> float:
        return self.cfg.initial_temp * (self.cfg.cooling_rate ** max(0, self.iteration - 1))

    # ----------------------- main update loop -----------------------------------------
    def step(self):
    # === One FFO iteration ===
    # 1) Evaluate current population
    # 2) Variation (crossover / local search) with temperature‑scaled moves
    # 3) Track the global best and update stagnation counters

        print(f"[FFO] iter={self.iteration:03d} starting... (T={self.cooling_temperature():.4f})", flush=True)
        fits = self.evaluate_agents()
        T = self.cooling_temperature()

        N = self.agents.shape[0]
        for i in range(N):
            if np.random.rand() < self.cfg.crossover_probability:
                j = np.random.randint(N)
                if j == i:
                    j = (j + 1) % N
                c1, c2 = self.crossover(self.agents[i], self.agents[j])
                f1 = self.obj(c1); f2 = self.obj(c2)
                self.eval_count += 2
                self.agents[i] = c1 if f1 < f2 else c2

            if np.random.rand() < self.cfg.mutation_probability:
                self.agents[i] = self.local_search(self.agents[i], T, i)

            if self.no_improve > self.cfg.stagnation_perturb_after:
                extra = self.no_improve - self.cfg.stagnation_perturb_after
                intensity = self.cfg.perturb_base + self.cfg.perturb_growth * max(0, extra)
                self.agents[i] = self.apply_perturbation(self.agents[i], intensity)

            self.agents[i] = np.clip(self.agents[i], self.lb, self.ub)

        self.cfg.step_size *= 0.99 if self.no_improve <= 50 else 0.98
        self.iteration += 1
        self.fitness_history.append(float(-self.best_fit))

    def should_terminate(self) -> bool:
        if self.iteration > self.cfg.max_iter:
            return True
        if self.no_improve > self.cfg.no_improve_limit:
            return True
        return False

    def run(self) -> FFOResult:
        t0 = time.time()
        while not self.should_terminate():
            self.step()
            if self.verbose:
                print(f"[FFO] iter={self.iteration-1:03d} best_acc={-self.best_fit:.4f} no_improve={self.no_improve} T={self.cooling_temperature():.4f}")
        wall = time.time() - t0
        return FFOResult(
            best_hparams={},
            best_val_accuracy=float(-self.best_fit),
            history_best_acc=self.fitness_history,
            eval_count=int(self.eval_count),
            wall_time_sec=float(wall),
        )


# --------------------------- CLI glue around FFO --------------------------------------

def make_cfg(mode: str, seed: int) -> FFOConfig:
    if mode == "test":
        return FFOConfig(
            num_agents=16,
            max_iter=8,
            no_improve_limit=6,
            crossover_probability=0.40,
            mutation_probability=0.70,
            step_size=0.25,
            initial_temp=1.0,
            cooling_rate=0.95,
            stagnation_perturb_after=5,
            perturb_base=0.05,
            perturb_growth=0.008,
            seed=seed,
        )
    elif mode == "full":
        return FFOConfig(
            num_agents=32,
            max_iter=20,
            no_improve_limit=30,
            crossover_probability=0.50,
            mutation_probability=0.70,
            step_size=0.30,
            initial_temp=1.0,
            cooling_rate=0.97,
            stagnation_perturb_after=12,
            perturb_base=0.06,
            perturb_growth=0.008,
            seed=seed,
        )
    else:
        raise ValueError("mode must be 'test' or 'full'")


def minimize_neg_accuracy(space: ParamSpace, arrays, cache_file: Path | None, fixed_epochs: int):
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

# --- Persistence ---
# Save config, space spec, best solution, and history so runs are reproducible
# and comparable across FFO/PSO/GWO. JSON is human‑readable for quick diffing.


def save_run(out_dir: Path, spec: SpaceSpec, cfg: FFOConfig, res: FFOResult, best_h: Dict[str, float | int], cache_file: Path | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "space.json").write_text(json.dumps(asdict(spec), indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "result.json").write_text(json.dumps({
        "best_hparams": best_h,
        "best_val_accuracy": res.best_val_accuracy,
        "history_best_acc": res.history_best_acc,
        "eval_count": res.eval_count,
        "wall_time_sec": res.wall_time_sec,
    }, indent=2))
    if cache_file and cache_file.exists():
        try:
            (out_dir / "objective_cache.json").write_text(cache_file.read_text())
        except Exception:
            pass


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper-faithful Firefighter Optimization for CNN hparam tuning")
    ap.add_argument("--mode", choices=["test", "full"], default="test")
    ap.add_argument("--data-dir", type=str, default=".")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=str, default=".ffo_cache.json")
    ap.add_argument("--subset-frac", type=float, default=None,
                    help="fraction of TRAIN set to use with stratified sampling (e.g., 0.25). If omitted: test mode defaults to 0.25; full mode uses 1.0")
    # fixed epochs per mode
    ap.add_argument("--epochs-fixed-test", type=int, default=3, help="fixed epochs used in test mode")
    ap.add_argument("--epochs-fixed-full", type=int, default=10, help="fixed epochs used in full mode")
    # search space overrides
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
    args = ap.parse_args(argv)

    set_global_seed(args.seed)

    spec = SpaceSpec(
        dense_units_min=args.dense_min, dense_units_max=args.dense_max,
        dropout_min=args.dropout_min, dropout_max=args.dropout_max,
        lr_min=args.lr_min, lr_max=args.lr_max,
        batch_min=args.batch_min, batch_max=args.batch_max,
        l2_min=args.l2_min, l2_max=args.l2_max,
    )
    space = ParamSpace(spec)

    arrays = load_arrays(Path(args.data_dir))
    cfg = make_cfg(args.mode, seed=args.seed)

    # Determine subset fraction: default 25% for test mode, full for thesis runs
    subset_frac = args.subset_frac
    if subset_frac is None:
        subset_frac = 0.25 if args.mode == "test" else None

    # If requested, stratified sample of the TRAIN split only (VAL stays full for fair eval)
    if subset_frac is not None and 0 < subset_frac < 1:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            train_X, train_y, val_X, val_y = arrays
            sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_frac, random_state=args.seed)
            (idx_small, _), = sss.split(train_X, train_y)
            train_X = train_X[idx_small]
            train_y = train_y[idx_small]
            arrays = (train_X, train_y, val_X, val_y)
            print(f"[FFO] Using subset of train: {len(idx_small)} samples (~{subset_frac*100:.0f}%)", flush=True)
        except Exception as e:
            print(f"[FFO] WARN: subset sampling failed ({e}); falling back to full train set.", flush=True)

    fixed_epochs = int(args.epochs_fixed_test if args.mode == "test" else args.epochs_fixed_full)
    print(f"[FFO] Fixed epochs this run: {fixed_epochs}", flush=True)

    cache_path = Path(args.cache) if args.cache else None
    objective = minimize_neg_accuracy(space, arrays, cache_path, fixed_epochs=fixed_epochs)

    engine = FirefighterOptimization(objective_minimize=objective, dim=space.dim, cfg=cfg, bounds=(0.0,1.0), seed=args.seed, verbose=True)

    print("\n[FFO] Config:\n" + json.dumps(asdict(cfg), indent=2), flush=True)
    print("[FFO] Search space:\n" + json.dumps(asdict(spec), indent=2), flush=True)

    res = engine.run()
    best_h = space.vec_to_hparams(engine.best_agent, fixed_epochs=fixed_epochs)

    print("\n[FFO] Best validation accuracy:", f"{res.best_val_accuracy:.4f}", flush=True)
    print("[FFO] Best hyperparameters:", flush=True)
    for k, v in best_h.items():
        print(f"  - {k}: {v}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / f"ffo_{args.mode}_{stamp}"
    save_run(run_dir, spec, cfg, res, best_h, cache_path)
    print(f"\n[FFO] Results saved to: {run_dir}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
