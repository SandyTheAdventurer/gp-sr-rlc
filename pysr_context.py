"""PySR (Cranmer, 2023) as context for the GP results (plans/REVISION_PLAN.md, Run 6). Not a competitor:
a complete SR system differs from our GP in almost every component.

Three steps:
    python pysr_context.py export                  # the GP runs' exact data -> results/pysr/tasks/
    .venv-pysr/bin/python pysr_context.py fit K    # PySR on every task, K worker processes
    python pysr_context.py score                   # test R², recovery (the GP's own check) -> results/pysr.csv

PySR 2.7.2 runs in its own environment (.venv-pysr), since Julia and PyTorch do not share a process well.
Settings: the same training data and function set as the GP ({+, -, *, /, sin, cos, exp, log}, plus sqrt for
Keijzer), 40 iterations, serial and deterministic with random_state = seed, PySR's defaults otherwise; PySR's
own model selection picks the final expression.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "pysr"
TASKS, FITS = OUT / "tasks", OUT / "fits"
SEEDS = range(10)
NITERATIONS = 40


def export():
    """Write each task's training and test data exactly as experiments.run_sr builds them."""
    import benchmarks
    import experiments as E
    TASKS.mkdir(parents=True, exist_ok=True)
    specs = E.all_specs()
    for k in E.ALL:
        spec = specs[k]
        names = [v["name"] for v in spec["variables"]]
        unary = E.STANDARD_UNARY[E.benchmark_of(k)]
        (TASKS / f"{k}.json").write_text(json.dumps({"names": names, "unary": unary}))
        for seed in SEEDS:
            X, y = E.sample(spec, np.random.default_rng(seed + k * 1000), benchmarks.SAMPLE_SIZE)
            Xt, yt = E.sample(spec, np.random.default_rng(10_000 + seed + k * 1000), benchmarks.SAMPLE_SIZE)
            np.savez(TASKS / f"{k}_{seed}.npz", X=X, y=y, X_test=Xt, y_test=yt)
    print(len(E.ALL) * len(SEEDS), "tasks ->", TASKS)


def fit_one(task):
    k, seed = task
    out = FITS / f"{k}_{seed}.json"
    if out.exists():
        return None
    from pysr import PySRRegressor
    d = np.load(TASKS / f"{k}_{seed}.npz")
    meta = json.loads((TASKS / f"{k}.json").read_text())
    model = PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=meta["unary"],
                          niterations=NITERATIONS, parallelism="serial", deterministic=True, random_state=seed,
                          progress=False, verbosity=0, temp_equation_file=True)
    t0 = time.time()
    model.fit(d["X"], d["y"], variable_names=meta["names"])
    elapsed = time.time() - t0
    best = model.get_best()
    with np.errstate(all="ignore"):
        pred = np.asarray(model.predict(d["X_test"]), dtype=float)
    yt = d["y_test"]
    r2 = float(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)) if np.all(np.isfinite(pred)) else float("nan")
    rec = {"spec_id": k, "seed": seed, "test_r2": r2, "complexity": int(best["complexity"]),
           "sympy": str(best["sympy_format"]), "elapsed_sec": round(elapsed, 2)}
    out.write_text(json.dumps(rec))
    return rec


def fit(jobs):
    FITS.mkdir(parents=True, exist_ok=True)
    tasks = [(k, s) for s in SEEDS for k in sorted(int(p.stem) for p in TASKS.glob("*.json"))]
    todo = [t for t in tasks if not (FITS / f"{t[0]}_{t[1]}.json").exists()]
    print(f"{len(todo)} fits to do ({len(tasks) - len(todo)} already done), {jobs} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs) as pool:
        for n, rec in enumerate(pool.imap_unordered(fit_one, todo), 1):
            eta = (time.time() - t0) / n * (len(todo) - n) / 60
            print(f"[{n}/{len(todo)}] eq={rec['spec_id']:3d} seed={rec['seed']} test_r2={rec['test_r2']:.4f} "
                  f"size={rec['complexity']} {rec['elapsed_sec']:.0f}s  eta {eta:.0f} min", flush=True)


def score():
    """Rows in the GP results format (arm "pysr"): recovery with the GP runs' own check, size = PySR's
    complexity (node count)."""
    import sympy as sp
    import experiments as E
    specs = E.all_specs()
    rows = []
    for f in sorted(FITS.glob("*.json")):
        o = json.loads(f.read_text())
        k, spec = o["spec_id"], specs[o["spec_id"]]
        names = json.loads((TASKS / f"{k}.json").read_text())["names"]
        row = {"arm": "pysr", "spec_id": k, "seed": o["seed"], "status": "ok", "error": "", "test_r2": o["test_r2"],
               "size": o["complexity"], "elapsed_sec": o["elapsed_sec"], "best_expr": o["sympy"],
               "evals_fitness": 0, "evals_selector": 0}
        try:
            model = sp.sympify(o["sympy"], locals={n: sp.Symbol(n) for n in names})
            row["recovered"] = int(E.Checker.call("is_recovered", model, spec["sympy"]))
        except Exception as exc:
            row.update(status="error", error=repr(exc))
        rows.append(row)
    with open(ROOT / "results" / "pysr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=E.SR_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", ROOT / "results" / "pysr.csv")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export()
    elif cmd == "fit":
        fit(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "score":
        score()
    else:
        raise SystemExit(__doc__)
