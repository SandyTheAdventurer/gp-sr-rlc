"""Nguyen + Keijzer benchmark: learned vs random crossover.

Usage: python run_benchmark.py [--jobs N] [--shard K/N] [--merge] [--benchmark nguyen|keijzer|all]
"""
from __future__ import annotations

import argparse
import csv
import glob
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmarks import SEEDS, Task, run_single_task, select_specs


# ============================================================================
# Reports
# ============================================================================

ROW_FIELDS = ["mode", "spec_id", "seed", "expr_target", "n_vars",
              "status", "best_r2", "ted", "best_expr", "elapsed_sec", "error"]
SUMMARY_FIELDS = [
    "mode", "spec_id", "expr_target", "input_range", "n_vars", "runs_ok", "runs_total",
    "r2_mean", "r2_median", "r2_std", "r2_min", "r2_max",
    "ted_mean", "ted_median", "ted_min",
    "time_mean_sec", "time_std_sec",
]


def _get_range_str(spec: Dict) -> str:
    """Format input range as compact string."""
    if not spec or "variables" not in spec:
        return "?"
    ranges = []
    for v in spec["variables"]:
        lo, hi = v["sampler"]
        ranges.append(f"[{lo:.0f},{hi:.0f}]" if lo == int(lo) and hi == int(hi) else f"[{lo:.1f},{hi:.1f}]")
    return ",".join(ranges)


def summarize_results(rows: List[Dict], specs: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for mode, sid in sorted({(r["mode"], r["spec_id"]) for r in rows}):
        sub = [r for r in rows if r["mode"] == mode and r["spec_id"] == sid and r["status"] == "ok"]
        r2 = np.array([r["best_r2"] for r in sub if np.isfinite(r["best_r2"])], dtype=float)
        ts = np.array([r["elapsed_sec"] for r in sub], dtype=float)
        ted_vals = np.array([r.get("ted", -1) for r in sub if r.get("ted", -1) >= 0], dtype=float)
        sp_ = next((s for s in specs if s["id"] == sid), {})
        f = lambda a, fn: float(fn(a)) if a.size else float("nan")
        out.append({"mode": mode, "spec_id": sid, "expr_target": sp_.get("expr", "?"),
                    "input_range": _get_range_str(sp_),
                    "n_vars": len(sp_.get("variables", [])), "runs_ok": len(sub),
                    "runs_total": sum(1 for r in rows if r["mode"] == mode and r["spec_id"] == sid),
                    "r2_mean": f(r2, np.mean), "r2_median": f(r2, np.median),
                    "r2_std": f(r2, np.std), "r2_min": f(r2, np.min), "r2_max": f(r2, np.max),
                    "ted_mean": f(ted_vals, np.mean), "ted_median": f(ted_vals, np.median),
                    "ted_min": f(ted_vals, np.min),
                    "time_mean_sec": f(ts, np.mean), "time_std_sec": f(ts, np.std)})
    return out


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> List[Dict]:
    """Read a run CSV written by write_csv, restoring the numeric columns."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["spec_id"], r["seed"], r["n_vars"] = int(r["spec_id"]), int(r["seed"]), int(r["n_vars"])
            r["best_r2"], r["elapsed_sec"] = float(r["best_r2"]), float(r["elapsed_sec"])
            r["ted"] = int(r.get("ted", -1)) if r.get("ted", "-1") not in ("", "-1", "nan") else -1
            rows.append(r)
    return rows


def write_summary_csv(rows: List[Dict], path: Path, specs: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader(); w.writerows(summarize_results(rows, specs))


def fmt_dur(s: float) -> str:
    if not np.isfinite(s): return "?"
    t = max(0, int(round(s))); h, r = divmod(t, 3600); m, s2 = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s2:02d}" if h else f"{m:02d}:{s2:02d}"


def print_comparison(rows: List[Dict], specs: List[Dict], bench_name: str = "NGUYEN") -> None:
    from scipy.stats import wilcoxon
    summary = summarize_results(rows, specs)
    sids = sorted({s["spec_id"] for s in summary})
    W = 160
    print("\n" + "=" * W + f"\n{bench_name} BENCHMARK — LEARNED vs RANDOM CROSSOVER\n" + "=" * W)
    lw = rw = ties = 0
    deltas: List[float] = []
    ted_deltas: List[float] = []
    print(f"{'Spec':>6} {'Expr':>20} {'Range':>18} {'#v':>3} {'L mean±std':>14} {'L med':>7}"
          f" {'R mean±std':>14} {'R med':>7} {'Delta':>8} {'L TED':>6} {'R TED':>6} {'L t/s':>7} {'R t/s':>7} {'Winner':>8}")
    print("-" * W)
    for sid in sids:
        lr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "learned"), None)
        rr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "random"), None)
        if not lr or not rr: continue
        d = lr["r2_mean"] - rr["r2_mean"]; deltas.append(d)
        # TED comparison (lower is better)
        l_ted = lr.get("ted_mean", float("nan"))
        r_ted = rr.get("ted_mean", float("nan"))
        if np.isfinite(l_ted) and np.isfinite(r_ted):
            ted_deltas.append(r_ted - l_ted)  # Positive = learned better
        if abs(d) < 1e-4: w, ties = "TIE", ties + 1
        elif d > 0: w = "LEARNED"; lw += 1
        else: w = "RANDOM"; rw += 1
        l_ted_str = f"{l_ted:>6.1f}" if np.isfinite(l_ted) else "   N/A"
        r_ted_str = f"{r_ted:>6.1f}" if np.isfinite(r_ted) else "   N/A"
        range_str = lr.get("input_range", "?")[:18]
        print(f"{sid:>6} {lr['expr_target'][:20]:>20} {range_str:>18} {lr['n_vars']:>3}"
              f" {lr['r2_mean']:>7.4f}±{lr['r2_std']:<5.3f} {lr['r2_median']:>7.4f}"
              f" {rr['r2_mean']:>7.4f}±{rr['r2_std']:<5.3f} {rr['r2_median']:>7.4f}"
              f" {d:>+8.4f} {l_ted_str} {r_ted_str} {lr['time_mean_sec']:>7.1f} {rr['time_mean_sec']:>7.1f} {w:>8}")
    print("-" * W)
    da = np.array(deltas) if deltas else np.array([])
    print(f"Learned wins: {lw}  |  Random wins: {rw}  |  Ties: {ties}")
    if da.size:
        print(f"Mean   R² delta: {float(np.mean(da)):+.4f}")
        print(f"Median R² delta: {float(np.median(da)):+.4f}")
    if da.size >= 6:
        try:
            stat, pv = wilcoxon(da, alternative="two-sided")
            ne = np.sum(da != 0)
            rb = 1.0 - (2.0 * stat) / (ne * (ne + 1) / 2) if ne > 0 else 0.0
            sig = " *" if pv < 0.05 else " (*)" if pv < 0.10 else ""
            print(f"Wilcoxon: W={stat:.1f}, p={pv:.4f}{sig}  |  Rank-biserial r={rb:+.3f}")
        except Exception as e:
            print(f"Wilcoxon failed: {e}")
    if da.size >= 2:
        sd = np.std(da, ddof=1)
        cd = float(np.mean(da) / sd) if sd > 0 else 0.0
        mg = "large" if abs(cd) >= 0.8 else "medium" if abs(cd) >= 0.5 else "small" if abs(cd) >= 0.2 else "negligible"
        print(f"Cohen's d: {cd:+.3f} ({mg})")
    # TED summary
    if ted_deltas:
        tda = np.array(ted_deltas)
        print(f"TED: Learned better in {sum(1 for t in tda if t > 0)}/{len(tda)} eqs  |  "
              f"Mean TED advantage: {float(np.mean(tda)):+.2f}")
    print(f"Equations: {len(sids)}  |  Seeds: {len(SEEDS)}\n" + "=" * W)


# ============================================================================
# Command line
# ============================================================================

def merge_shards(args, specs, bench_name) -> None:
    shard_pattern = f"{args.benchmark}_results_shard*.csv"
    files = sorted(glob.glob(str(args.output.parent / "shards" / shard_pattern)))
    if not files:
        raise SystemExit(f"No shard files found matching {shard_pattern}")
    all_rows: List[Dict] = []
    for p in files:
        print(f"Reading {p} ...", flush=True)
        all_rows.extend(read_csv(Path(p)))
    all_rows.sort(key=lambda r: (r["mode"], r["spec_id"], r["seed"]))
    d = args.output.parent
    write_csv(all_rows, d / f"{args.benchmark}_results.csv")
    write_summary_csv(all_rows, d / f"{args.benchmark}_results_summary.csv", specs)
    print(f"\nMerged {len(all_rows)} rows from {len(files)} shards")
    print_comparison(all_rows, specs, bench_name)


def main() -> None:
    pa = argparse.ArgumentParser(description="Nguyen + Keijzer benchmark")
    pa.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 1))
    pa.add_argument("--output", type=Path, default=None, help="Output CSV (default: {benchmark}_results.csv)")
    pa.add_argument("--summary-output", type=Path, default=None)
    pa.add_argument("--progress-every", type=int, default=5)
    pa.add_argument("--shard", type=str, default=None, help="K/N (e.g. 1/4)")
    pa.add_argument("--merge", action="store_true", help="Merge shard CSVs")
    pa.add_argument("--benchmark", type=str, default="all", choices=["nguyen", "keijzer", "all"],
                    help="Which benchmark to run (default: all)")
    args = pa.parse_args()

    specs_all, bench_name = select_specs(args.benchmark)

    # Set default output paths based on benchmark
    if args.output is None:
        args.output = Path(f"{args.benchmark}_results.csv")
    if args.summary_output is None:
        args.summary_output = Path(f"{args.benchmark}_results_summary.csv")

    if args.merge:
        merge_shards(args, specs_all, bench_name)
        return

    shard_idx = num_shards = None
    spec_sub: List[Dict] | None = None
    if args.shard:
        try:
            k, n = args.shard.split("/"); shard_idx, num_shards = int(k), int(n)
        except (ValueError, AttributeError):
            raise SystemExit(f"ERROR: --shard must be K/N, got '{args.shard}'")
        if not (1 <= shard_idx <= num_shards):
            raise SystemExit(f"ERROR: shard {shard_idx} out of 1..{num_shards}")
        ss = sorted(specs_all, key=lambda s: s["id"])
        spec_sub = [s for i, s in enumerate(ss) if i % num_shards == shard_idx - 1]
        if not spec_sub:
            raise SystemExit(f"Shard {shard_idx}/{num_shards} is empty.")

    specs = spec_sub or specs_all
    tasks = [Task(m, s["id"], sd, args.benchmark) for m in ("random", "learned") for s in specs for sd in SEEDS]
    nt = len(tasks)

    if shard_idx:
        out_csv = args.output.with_name(f"{args.output.stem}_shard{shard_idx}.csv")
        sum_csv = args.summary_output.with_name(f"{args.summary_output.stem}_shard{shard_idx}.csv")
        tag = f"  [shard {shard_idx}/{num_shards}]"
    else:
        out_csv, sum_csv, tag = args.output, args.summary_output, ""

    jobs = max(1, min(args.jobs, nt))
    print(f"{bench_name}{tag}: {len(specs)} eq × {len(SEEDS)} seeds × 2 = {nt} tasks | Workers: {jobs}")
    print(f"IDs: {sorted(s['id'] for s in specs)}")

    t0 = time.time(); rows: List[Dict] = []; mc = {"random": 0, "learned": 0}; errs = 0
    pf = max(1, args.progress_every)
    try:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for done, row in enumerate(pool.imap_unordered(run_single_task, tasks), 1):
                rows.append(row); m = row["mode"]; mc[m] += 1
                if row["status"] != "ok": errs += 1
                dt = time.time() - t0; rate = done / dt if dt > 0 else 0; eta = (nt - done) / rate if rate > 0 else float("inf")
                if done == 1 or done == nt or done % pf == 0 or row["status"] != "ok":
                    r2t = f"{row['best_r2']:.4f}" if np.isfinite(row["best_r2"]) else "nan"
                    print(f"[{done}/{nt}] {m} spec={row['spec_id']} seed={row['seed']} "
                          f"{row['status']} r2={r2t} {row['elapsed_sec']:.1f}s errs={errs} "
                          f"eta={fmt_dur(eta)}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")

    rows.sort(key=lambda r: (r["mode"], r["spec_id"], r["seed"]))
    write_csv(rows, out_csv); write_summary_csv(rows, sum_csv, specs_all)
    print(f"\nDone in {fmt_dur(time.time() - t0)}. {len(rows)} rows -> {out_csv} | Errors: {sum(1 for r in rows if r['status'] != 'ok')}")
    print_comparison(rows, specs_all, bench_name)


if __name__ == "__main__":
    main()
