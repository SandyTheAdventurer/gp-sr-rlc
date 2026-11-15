"""The revision's experiments (plans/REVISION_PLAN.md).

Symbolic regression: each run trains on one equation with one seed and one arm (crossover variant)
and records train and held-out test R², exact symbolic recovery, tree edit distance to the true
equation, model size, wall time, fitness evaluations and the per-generation curve.

Resistor networks: each run evolves a network for one target resistance with one seed and one
crossover arm, and records the smallest relative error on the final Pareto front.

Rows are appended as runs finish, so an interrupted run resumes where it stopped.

Usage:
    python experiments.py sr --equations heldout --arms random,full --seeds 30 --jobs 8 --tag main
    python experiments.py sr ... --noise 0.01 --tag noise01
    python experiments.py sr --equations nguyen_heldout ... --protocol orig --tag nguyen_orig
    python experiments.py circuits --arms swap,learned,uniform --targets 30 --seeds 10 --jobs 8
"""
from __future__ import annotations

import os

# One thread per run, CPU only, no progress bars. Set before numpy and torch load; values already
# in the environment win
for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "TQDM_DISABLE": "1", "CUDA_VISIBLE_DEVICES": "", "GP_SR_RLC_SELECTOR_DEVICE": "cpu"}.items():
    os.environ.setdefault(_k, _v)

import argparse
import atexit
import csv
import json
import math
import multiprocessing as mp
import pickle
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sympy as sp  # torch loads in the run functions, so the recovery-check helper stays small

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results"

# ============================================================================
# Arms
# ============================================================================

# Every arm starts from the paper's configuration (benchmarks.ga_config + learned_overrides:
# population 300, 200 generations, Pareto selection, warmup 30, ...) and changes only what is listed.
RANDOM = {"use_learned_crossover": False}
FULL = {}  # GP-SR-RLC as published: the learned policy plus the structural heuristic
ARMS = {
    "random": RANDOM,
    "full": FULL,
    # Ablations (R4.2, R5.4). With the heuristic off, the pair softmax over l_i + l'_j factorizes into two
    # independent node choices, so "policy" is also the independent-sampling arm.
    "policy": {"heuristic_bias_scale": 0.0},
    "heuristic": {"crossover_operator": "heuristic"},
    "nocritic": {"use_critic": False},
    "full_noconst": {"opt_constants": False},
    "random_noconst": {**RANDOM, "opt_constants": False},
    "full_m01": {"mutation_rate": 0.1},
    "random_m01": {**RANDOM, "mutation_rate": 0.1},
    # Informed baselines (R4.1, R5.1), acting from generation 1 like random crossover
    "ssc": {"crossover_operator": "ssc", "selector_warmup_gens": 1},
    "sizefair": {"crossover_operator": "sizefair", "selector_warmup_gens": 1},
}
# Equal compute (R5.5): random crossover with K times the generations. K is fixed in the plan.
for _k in (2, 3, 4, 5, 6, 8):
    ARMS[f"random_x{_k}"] = {**RANDOM, "gens": 200 * _k}

SR_FIELDS = ["arm", "spec_id", "seed", "status", "train_r2", "test_r2", "test_r2_orig", "recovered", "ted", "ned",
             "size", "gens_used", "elapsed_sec", "evals_fitness", "evals_selector", "best_expr", "error"]

# ============================================================================
# Equations, function sets and data
# ============================================================================

HELDOUT = [1, 2, 3, 4, 5, 8, 9, 10, 11, *range(101, 114), 115]  # the 23 held-out Nguyen + Keijzer equations
DEV = [6, 7, 12, 114]  # used for development in the first revision; used here only for smoke tests
GROUPS = {"heldout": HELDOUT, "dev": DEV, "nguyen_heldout": [k for k in HELDOUT if k < 100],
          "keijzer_heldout": [k for k in HELDOUT if k > 100]}

# "standard": each benchmark's own function set for every equation (Nguyen: Uy et al., 2011,
# {+, -, *, /, sin, cos, exp, log}; Keijzer: the same plus sqrt), no pow.
# "paper": the published runs' per-equation sets (only the functions in the true equation, pow for N11/K10)
NGUYEN_UNARY = ["sin", "cos", "exp", "log"]
STANDARD_UNARY = {"nguyen": NGUYEN_UNARY, "keijzer": NGUYEN_UNARY + ["sqrt"]}

# The original Nguyen protocol (Uy et al., 2011, as tabulated by Petersen et al., 2021): 20 training points
# from these ranges, instead of 512 from the wider ranges in benchmarks.py
NGUYEN_ORIG_RANGES = {**{k: [(-1.0, 1.0)] for k in range(1, 7)}, 7: [(0.0, 2.0)], 8: [(0.0, 4.0)],
                      **{k: [(0.0, 1.0), (0.0, 1.0)] for k in range(9, 13)}}
NGUYEN_ORIG_POINTS = 20


def benchmark_of(spec_id):
    return "nguyen" if spec_id < 100 else "keijzer"


def all_specs():
    import benchmarks
    return {**{s["id"]: s for s in benchmarks.NGUYEN_SPECS}, **{s["id"] + 100: s for s in benchmarks.KEIJZER_SPECS}}


def sample(spec, rng, n):
    cols = [rng.uniform(*v["sampler"], size=n) for v in spec["variables"]]
    with np.errstate(all="ignore"):
        y = np.asarray(spec["func"](*cols), dtype=float).ravel()
    ok = np.isfinite(y)
    return np.column_stack(cols)[ok], y[ok]


def _grid(*axes):
    """Every combination of the given 1-D point sets, one column per variable."""
    return np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])


def _even(a, b, step):
    return np.linspace(a, b, int(round((b - a) / step)) + 1)


def keijzer_test_set(k):
    """Keijzer's (2003) test set for Keijzer-k: mostly a dense grid over the training range; K4 is shifted
    and K6 extends from x <= 50 to x <= 120, so those two extrapolate."""
    E = _even
    if k in (1, 2, 3):
        return _grid(E(-k, k, 0.001))
    if k == 4:
        return _grid(E(0.05, 10.05, 0.05))
    if k == 5:
        rng = np.random.default_rng(5)
        return np.column_stack([rng.uniform(-1, 1, 10000), rng.uniform(1, 2, 10000), rng.uniform(-1, 1, 10000)])
    if k == 6:
        return _grid(E(1, 120, 1))
    if k == 7:
        return _grid(E(1, 100, 0.1))
    if k in (8, 9):
        return _grid(E(0, 100, 0.1))
    if k == 10:
        return _grid(E(0, 1, 0.01), E(0, 1, 0.01))
    return _grid(E(-3, 3, 0.01), E(-3, 3, 0.01))  # 11-15


def r2_of(y, pred):
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))

# ============================================================================
# Exact recovery and tree edit distance
# ============================================================================

# Small rationals times a nonzero power of e or pi (e.g. exp(-1), pi/2), either sign: function sets
# with exp and log produce these, which no nearby rational can stand for
_SPECIAL = [sign * q * base ** k for q in (1, 2, 3, 4, sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4))
            for base in (sp.E, sp.pi) for k in (-2, -1, 1, 2) for sign in (1, -1)]


def snap_constant(f, candidates, rel_tol):
    """The first candidate within rel_tol (relative) of f, else the nearby simple rational (within 1e-3)."""
    for c in candidates:
        if abs(float(f) - float(c)) <= rel_tol * abs(float(c)):
            return c
    return sp.nsimplify(f, rational=True, tolerance=1e-3)


def snap_constants(model, target):
    """Model with real-valued variables and evolved constants snapped: within 0.1% of a float constant
    of the target to that (Keijzer-6's Euler-Mascheroni constant), within 0.01% of a small rational
    times a power of e or pi to that, else to a nearby simple rational (within 1e-3)."""
    model = model.xreplace({x: sp.Symbol(x.name, real=True) for x in model.free_symbols})
    own = [sign * c for c in target.atoms(sp.Float) for sign in (1, -1)]
    return model.xreplace({f: snap_constant(f, own, 1e-3) if any(abs(float(f) - float(c)) <= 1e-3 * abs(float(c))
                                                                for c in own) else snap_constant(f, _SPECIAL, 1e-4)
                           for f in model.atoms(sp.Float)})


def is_recovered(model, target):
    """Exact symbolic recovery: model - target simplifies to 0 once evolved constants are snapped."""
    try:
        model = snap_constants(model, target)
        target = target.xreplace({x: sp.Symbol(x.name, real=True) for x in target.free_symbols})
        diff = model - target
        # Merge logs (log(a) - log(b) -> log(a/b)), which plain simplify won't do without sign assumptions
        return sp.simplify(diff) == 0 or sp.simplify(sp.logcombine(sp.expand_log(diff, force=True), force=True)) == 0
    except Exception:
        return False


def _zss_tree(expr):
    """zss tree of a sympy expression: operator class names, variable names, numbers as 'const'."""
    import zss
    if expr.is_Number or expr.is_NumberSymbol:
        return zss.Node("const")
    if expr.is_Symbol:
        return zss.Node(str(expr))
    node = zss.Node(type(expr).__name__)
    for arg in expr.args:
        node.addkid(_zss_tree(arg))
    return node


def tree_edit_distance(model, target):
    """(TED, NED): Zhang-Shasha tree edit distance between the model (constants snapped as for recovery)
    and the true equation, both simplified so they share a canonical form, and NED = min(1, TED / true
    tree size) as in SRSD (Matsubara et al., 2024)."""
    import zss
    exprs = [sp.simplify(e) for e in (snap_constants(model, target), snap_constants(target, target))]
    ted = int(zss.simple_distance(_zss_tree(exprs[0]), _zss_tree(exprs[1])))
    return ted, float(min(1.0, ted / sum(1 for _ in sp.preorder_traversal(exprs[1]))))


class Checker:
    """is_recovered and tree_edit_distance in a helper process, killed after a hard time limit.

    SymPy can get stuck in one long C call (a huge integer power) that no in-process alarm can interrupt;
    in the first revision 7 of 11,340 runs hung there. The helper starts once per worker and is reused. A
    killed recovery check counts as not recovered; a killed TED is NaN. The helper runs with
    PYTHONHASHSEED=0, because SymPy's results can depend on the order of hashed arguments.
    """
    LIMITS = {"is_recovered": (60, False), "tree_edit_distance": (150, (float("nan"), float("nan")))}
    proc = None

    @classmethod
    def call(cls, name, *args):
        limit, default = cls.LIMITS[name]
        if cls.proc is None or cls.proc.poll() is not None:
            cls.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--checker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env={**os.environ, "PYTHONHASHSEED": "0"})
            atexit.register(cls.proc.kill)
            if cls._recv(600) != ("ok", "ready"):
                raise RuntimeError("the recovery-check helper did not start")
        cls._send(cls.proc.stdin, (name, args))
        res = cls._recv(limit)
        if res is None:  # over the limit, or the helper died
            cls.proc.kill()
            cls.proc.wait()
            cls.proc = None
            return default
        if res[0] == "exc":
            raise res[1]
        return res[1]

    @staticmethod
    def _send(f, obj):
        b = pickle.dumps(obj)
        f.write(len(b).to_bytes(8, "little") + b)
        f.flush()

    @staticmethod
    def _read(f):
        h = f.read(8)
        if len(h) < 8:
            raise EOFError
        return pickle.loads(f.read(int.from_bytes(h, "little")))

    @classmethod
    def _recv(cls, timeout):
        """The helper's next reply, or None if none arrives within timeout seconds."""
        box = []

        def read(f=cls.proc.stdout):
            try:
                box.append(cls._read(f))
            except Exception:
                pass
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        return box[0] if box else None


def checker_main():
    """The helper's loop: read (name, args), reply ("ok", result) or ("exc", exception)."""
    limit_memory(3 << 30)
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # keep stray prints off the reply pipe
    Checker._send(out, ("ok", "ready"))
    funcs = {"is_recovered": is_recovered, "tree_edit_distance": tree_edit_distance}
    while True:
        try:
            name, args = Checker._read(inp)
        except EOFError:
            return
        try:
            res = ("ok", funcs[name](*args))
        except Exception as exc:
            res = ("exc", exc)
        try:
            Checker._send(out, res)
        except Exception:
            Checker._send(out, ("exc", RuntimeError(repr(res[1]))))


def limit_memory(nbytes):
    """Make a run that allocates more than nbytes fail on its own (status error) instead of pushing the
    machine into swap (Linux and macOS)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_DATA, (nbytes, nbytes))
    except (ImportError, ValueError, OSError):
        pass

# ============================================================================
# One symbolic-regression run
# ============================================================================


def run_sr(task):
    arm, spec_id, seed, opts = task
    import torch
    torch.set_num_threads(1)
    limit_memory(opts["memory_gb"] << 30)
    import benchmarks
    from gp_sr_rlc import crossover as lx, gp, run_ga
    spec = all_specs()[spec_id]
    row = {"arm": arm, "spec_id": spec_id, "seed": seed, "status": "ok", "error": ""}
    curve, counts = [], {"fitness": 0, "selector": 0}

    # Count fitness evaluations (each scores one tree on the full training set) without changing them:
    # population scoring in run_ga, and the selector's reward computation
    def counted(f, key):
        def wrapper(*a, **k):
            counts[key] += 1
            return f(*a, **k)
        return wrapper
    gp_score, lx_fitness = gp.score_individual, lx.fitness
    gp.score_individual, lx.fitness = counted(gp_score, "fitness"), counted(lx_fitness, "selector")
    t0 = time.time()
    try:
        n_train = benchmarks.SAMPLE_SIZE
        if opts["protocol"] == "orig" and benchmark_of(spec_id) == "nguyen":
            spec = {**spec, "variables": [{**v, "sampler": r} for v, r in zip(spec["variables"], NGUYEN_ORIG_RANGES[spec_id])]}
            n_train = NGUYEN_ORIG_POINTS
        # The training sample is the published runner's (benchmarks.run_single_task); the test sample is
        # fresh points from the same ranges
        X, y = sample(spec, np.random.default_rng(seed + spec_id * 1000), n_train)
        if opts["noise"]:
            # Gaussian noise on the training targets only, sd = noise * RMS(y) as in SRBench; the test set stays clean
            y = y + np.random.default_rng(20_000 + seed + spec_id * 1000).normal(
                0.0, opts["noise"] * np.sqrt(np.mean(y ** 2)), len(y))
        X_test, y_test = sample(spec, np.random.default_rng(10_000 + seed + spec_id * 1000), benchmarks.SAMPLE_SIZE)
        names = [v["name"] for v in spec["variables"]]
        if opts["functions"] == "standard":
            unary = {n: benchmarks.UNARY_REGISTRY[n] for n in STANDARD_UNARY[benchmark_of(spec_id)]}
            allow_pow = False
        else:
            unary = {n: benchmarks.UNARY_REGISTRY[n] for n in spec.get("allowed_unary") or [] if n in benchmarks.UNARY_REGISTRY}
            allow_pow = bool(spec.get("allow_pow", False))
        kw = {**benchmarks.ga_config, **benchmarks.learned_overrides, **ARMS[arm], "verbose": False, "seed": seed,
              "allow_pow": allow_pow,
              "metrics_callback": lambda gen, r2: curve.append(
                  [round(r2, 6), round(time.time() - t0, 3), counts["fitness"] + counts["selector"]])}
        best, train_r2 = run_ga(X, y, var_names=names, unary_funcs=unary, **kw)
        with np.errstate(all="ignore"):
            pred = best.evaluate({n: X_test[:, i] for i, n in enumerate(names)})
        row.update(train_r2=float("nan") if train_r2 is None else float(train_r2), test_r2=r2_of(y_test, pred),
                   size=best.size(), gens_used=len(curve), best_expr=best.to_string(),
                   evals_fitness=counts["fitness"], evals_selector=counts["selector"])
        if benchmark_of(spec_id) == "keijzer":
            Xk = keijzer_test_set(spec_id - 100)
            with np.errstate(all="ignore"):
                yk = np.asarray(spec["func"](*Xk.T), dtype=float)
                pk = best.evaluate({n: Xk[:, i] for i, n in enumerate(names)})
            row["test_r2_orig"] = r2_of(yk, pk)
        row["elapsed_sec"] = round(time.time() - t0, 2)  # GA time only, excludes the recovery check
        model = best.to_sympy()
        row["recovered"] = int(Checker.call("is_recovered", model, spec["sympy"]))
        # A recovered model is the true equation, whatever form simplify leaves it in
        row["ted"], row["ned"] = (0, 0.0) if row["recovered"] else Checker.call("tree_edit_distance", model, spec["sympy"])
    except Exception as exc:
        row.update(status="error", error=repr(exc), elapsed_sec=round(time.time() - t0, 2))
    finally:
        gp.score_individual, lx.fitness = gp_score, lx_fitness
    return row, curve

# ============================================================================
# One resistor-network run
# ============================================================================

E6 = [m * 10 ** k for k in range(3) for m in (10, 15, 22, 33, 47, 68)]  # 10 ohm to 6.8 kohm
CIRCUIT = {"nodes": 6, "nr": 6, "pop_size": 100, "gens": 50}
CIRCUIT_ARMS = {"swap": {"swap": True}, "learned": {"learned": True}, "uniform": {}}
CIRCUIT_FIELDS = ["arm", "target", "seed", "best_error", "best_resistors", "solved", "front_size", "evaluations",
                  "elapsed_sec"]
SOLVED = 1e-4  # relative error


def circuit_targets(n):
    """n targets, log-uniform between 5 ohm and 5 kohm, fixed by seed 123."""
    return [round(float(t), 2) for t in np.exp(np.random.default_rng(123).uniform(math.log(5), math.log(5000), n))]


def run_circuit(task):
    arm, target, seed = task
    import torch
    torch.set_num_threads(1)
    import circuits
    circuits.R_VALUES[:] = E6
    t0 = time.time()
    stats = {}
    _, _, front = circuits.evolve(target=target, seed=seed, verbose=False, stats=stats, **CIRCUIT, **CIRCUIT_ARMS[arm])
    objs = sorted(circuits.evaluate_pareto(t, target) for t in front)
    err, nres = objs[0]
    return {"arm": arm, "target": target, "seed": seed, "best_error": float(err), "best_resistors": nres,
            "solved": int(err < SOLVED), "front_size": len(front), "evaluations": stats["evaluations"],
            "elapsed_sec": round(time.time() - t0, 2)}

# ============================================================================
# Command line
# ============================================================================


def run_pool(func, tasks, out_csv, fields, jobs, describe, curves_path=None):
    new_file = not out_csv.exists()
    t0 = time.time()
    fj = open(curves_path, "a", encoding="utf-8") if curves_path else None
    with open(out_csv, "a", newline="", encoding="utf-8") as fc, \
            mp.get_context("spawn").Pool(jobs, maxtasksperchild=12) as pool:
        writer = csv.DictWriter(fc, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for k, result in enumerate(pool.imap_unordered(func, tasks), 1):
            row, curve = result if isinstance(result, tuple) else (result, None)
            writer.writerow(row)
            fc.flush()
            if fj is not None:
                fj.write(json.dumps({"arm": row["arm"], "spec_id": row["spec_id"], "seed": row["seed"], "curve": curve}) + "\n")
                fj.flush()
            eta = (time.time() - t0) / k * (len(tasks) - k)
            print(f"[{k}/{len(tasks)}] {describe(row)}  eta {eta / 60:.0f} min", flush=True)
    if fj is not None:
        fj.close()


def main():
    if sys.argv[1:] == ["--checker"]:
        return checker_main()
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = pa.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sr", help="symbolic-regression runs")
    s.add_argument("--equations", default="dev", help="spec ids (Keijzer offset by 100) or groups: " + ", ".join(GROUPS))
    s.add_argument("--arms", default="random,full")
    s.add_argument("--seeds", type=int, default=30)
    s.add_argument("--seed0", type=int, default=0)
    s.add_argument("--noise", type=float, default=0.0, help="Gaussian noise on training targets, sd = noise * RMS(y)")
    s.add_argument("--protocol", choices=["paper", "orig"], default="paper",
                   help="orig: Nguyen with the original 20 points and ranges")
    s.add_argument("--functions", choices=["standard", "paper"], default="standard")
    s.add_argument("--gens", type=int, default=None, help="override generations (smoke tests only)")
    s.add_argument("--memory-gb", type=int, default=3, help="per-run memory limit")
    s.add_argument("--jobs", type=int, default=8)
    s.add_argument("--tag", required=True, help="output file name: results/<tag>.csv")
    c = sub.add_parser("circuits", help="resistor-network runs")
    c.add_argument("--arms", default="swap,learned,uniform")
    c.add_argument("--targets", type=int, default=30)
    c.add_argument("--seeds", type=int, default=10)
    c.add_argument("--jobs", type=int, default=8)
    c.add_argument("--tag", default="circuits")
    args = pa.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{args.tag}.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            key = ("spec_id", int) if args.cmd == "sr" else ("target", float)
            done = {(r["arm"], key[1](r[key[0]]), int(r["seed"])) for r in csv.DictReader(f)}

    if args.cmd == "sr":
        opts ={"noise": args.noise, "protocol": args.protocol, "functions": args.functions, "memory_gb": args.memory_gb}
        equations = [e for tok in args.equations.split(",") for e in (GROUPS[tok] if tok in GROUPS else [int(tok)])]
        arms = args.arms.split(",")
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}; choose from {list(ARMS)}")
        tasks = [(a, e, sd, opts) for sd in range(args.seed0, args.seed0 + args.seeds) for e in equations for a in arms
                 if (a, e, sd) not in done]
        if args.gens is not None:  # the override must reach the spawned workers, which re-import ARMS
            tasks = [(a, e, sd, {**o, "gens": args.gens}) for a, e, sd, o in tasks]
        print(f"{len(tasks)} runs to do ({len(done)} already done), {args.jobs} workers", flush=True)
        run_pool(run_sr_task, tasks, out_csv, SR_FIELDS, args.jobs,
                 lambda r: f"{r['arm']:14s} eq={r['spec_id']:3d} seed={r['seed']:2d} {r['status']} "
                           f"test_r2={r.get('test_r2', float('nan')):.4f} rec={r.get('recovered', '-')} {r['elapsed_sec']:.0f}s",
                 curves_path=OUT_DIR / f"{args.tag}_curves.jsonl")
    else:
        tasks = [(a, t, sd) for sd in range(args.seeds) for t in circuit_targets(args.targets)
                 for a in args.arms.split(",") if (a, t, sd) not in done]
        print(f"{len(tasks)} runs to do ({len(done)} already done), {args.jobs} workers", flush=True)
        run_pool(run_circuit, tasks, out_csv, CIRCUIT_FIELDS, args.jobs,
                 lambda r: f"{r['arm']:8s} target={r['target']:8.2f} seed={r['seed']:2d} error={r['best_error']:.2e}")


def run_sr_task(task):
    """run_sr with the smoke-test generation override applied in the worker."""
    arm, spec_id, seed, opts = task
    if "gens" in opts:
        k = int(arm[len("random_x"):]) if arm.startswith("random_x") else 1
        ARMS[arm] = {**ARMS[arm], "gens": opts["gens"] * k}
    return run_sr(task)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
