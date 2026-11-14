"""Analysis of the revision's runs, exactly as plans/REVISION_PLAN.md specifies.

Written before any of the planned runs. Each command reads one results CSV and prints its tests;
`figures` writes the box plots and convergence curves to results/figs/.

Usage:
    python analysis.py main      [results/main.csv]
    python analysis.py noise     results/noise01.csv results/noise10.csv
    python analysis.py protocol  [results/nguyen_orig.csv]
    python analysis.py circuits  [results/circuits.csv]
    python analysis.py figures   [results/main.csv]
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOLVED_R2 = 0.999
EQUAL_COMPUTE = "random_x3"  # fixed in the plan from the dev-equation timing pilot

# ============================================================================
# Loading and per-equation scores
# ============================================================================


def load(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    errors = [r for r in rows if r.get("status", "ok") != "ok"]
    if errors:
        print(f"{len(errors)} error rows in {path}:")
        for r in errors:
            print(f"  {r['arm']} eq={r.get('spec_id', r.get('target'))} seed={r['seed']}: {r['error'][:120]}")
    return rows


def score(r, field="test_r2"):
    """A run's R²: negative, non-finite, missing or error counts as 0."""
    if r["status"] != "ok":
        return 0.0
    try:
        v = float(r[field])
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def per_equation(rows, arm, stat):
    """{spec_id: stat over the arm's seeds on that equation}."""
    by = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            by[int(r["spec_id"])].append(r)
    return {k: stat(v) for k, v in by.items()}


def median_r2(field="test_r2"):
    return lambda runs: float(np.median([score(r, field) for r in runs]))


def solved_fraction(runs):
    return float(np.mean([score(r) >= SOLVED_R2 for r in runs]))


def recovery_rate(runs):
    return float(np.mean([r["status"] == "ok" and r["recovered"] == "1" for r in runs]))

# ============================================================================
# Tests
# ============================================================================


def paired(rows, a, b, stat=median_r2(), name="median test R²"):
    """Two-sided Wilcoxon signed-rank over equations of a's and b's per-equation values, zero differences
    dropped. Returns (label, wins, losses, ties, p)."""
    va, vb = per_equation(rows, a, stat), per_equation(rows, b, stat)
    eqs = sorted(set(va) & set(vb))
    d = np.array([va[k] - vb[k] for k in eqs])
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if (d != 0).any() else 1.0
    return f"{a} vs {b} ({name})", w, l, t, p


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def print_family(title, results):
    print(f"\n{title} (Holm-corrected as one family)")
    adj = holm([r[4] for r in results])
    for (label, w, l, t, p), pa in zip(results, adj):
        print(f"  {label:58s} {w:2d}W/{l:2d}L/{t:2d}T  p={p:.3g}  Holm p={pa:.3g}")


def print_test(title, result):
    label, w, l, t, p = result
    print(f"\n{title}\n  {label:58s} {w:2d}W/{l:2d}L/{t:2d}T  p={p:.3g}")


def a12(x, y):
    """Vargha-Delaney A: probability that a run of x beats a run of y (ties count half)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def per_equation_tests(rows, a, b):
    """Per equation: Mann-Whitney U on the 30 runs' test R² (scored as above), Holm over the equations,
    with the Vargha-Delaney effect size A12."""
    ra, rb = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["arm"] == a:
            ra[int(r["spec_id"])].append(score(r))
        elif r["arm"] == b:
            rb[int(r["spec_id"])].append(score(r))
    eqs = sorted(set(ra) & set(rb))
    ps = [float(mannwhitneyu(ra[k], rb[k], alternative="two-sided").pvalue) if len(set(ra[k] + rb[k])) > 1 else 1.0
          for k in eqs]
    print(f"\nPer equation: {a} vs {b}, Mann-Whitney U on test R², Holm over {len(eqs)} equations")
    print(f"  {'eq':>4s} {a + ' med':>12s} {b + ' med':>12s} {'A12':>6s} {'p':>8s} {'Holm p':>8s}")
    for k, p, pa in zip(eqs, ps, holm(ps)):
        print(f"  {label_of(k):>4s} {np.median(ra[k]):12.4f} {np.median(rb[k]):12.4f} {a12(ra[k], rb[k]):6.3f} {p:8.3g} {pa:8.3g}")


def label_of(spec_id):
    return f"N{spec_id}" if spec_id < 100 else f"K{spec_id - 100}"


def describe(rows, arms):
    """Per arm, over all its runs: mean and median test R², solved fraction, recovery rate, median size,
    mean time and fitness evaluations."""
    print(f"\n{'arm':16s} {'runs':>5s} {'mean R²':>8s} {'med R²':>8s} {'solved':>7s} {'recov':>6s} {'size':>5s} "
          f"{'time s':>7s} {'evals':>9s}")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        sc = [score(r) for r in rs]
        evals = [int(r["evals_fitness"]) + int(r["evals_selector"]) for r in ok]
        print(f"{arm:16s} {len(rs):5d} {np.mean(sc):8.4f} {np.median(sc):8.4f} {solved_fraction(rs):7.3f} "
              f"{recovery_rate(rs):6.3f} {np.median([int(r['size']) for r in ok]):5.0f} "
              f"{np.mean([float(r['elapsed_sec']) for r in ok]):7.1f} {np.mean(evals):9.0f}")

# ============================================================================
# Runs
# ============================================================================


def analyse_main(path):
    rows = load(path)
    arms = ["random", "full", "policy", "heuristic", "nocritic", "full_noconst", "random_noconst", "full_m01",
            "random_m01", "ssc", "sizefair", EQUAL_COMPUTE]
    print_test("Primary test", paired(rows, "full", "random"))
    print_family("Secondary tests", [
        paired(rows, "full", "random", solved_fraction, "solved fraction"),
        paired(rows, "full", "random", recovery_rate, "recovery rate"),
        paired(rows, "full", "heuristic"),
        paired(rows, "full", "policy"),
        paired(rows, "policy", "random"),
        paired(rows, "heuristic", "random"),
        paired(rows, "full", "nocritic"),
        paired(rows, "full_noconst", "random_noconst"),
        paired(rows, "full_m01", "random_m01"),
        paired(rows, "full", "ssc"),
        paired(rows, "full", "sizefair"),
        paired(rows, "ssc", "random"),
        paired(rows, "sizefair", "random"),
        paired(rows, "full", EQUAL_COMPUTE),
    ])
    print("\nDescriptive")
    describe(rows, arms)
    per_equation_tests(rows, "full", "random")
    kz = [r for r in rows if int(r["spec_id"]) > 100]
    print_test("Descriptive: Keijzer's original test sets (14 equations)",
               paired(kz, "full", "random", median_r2("test_r2_orig"), "median R² on Keijzer's test set"))
    for k in (104, 106):  # the two Keijzer test sets that extrapolate
        a = [score(r, "test_r2_orig") for r in kz if r["arm"] == "full" and int(r["spec_id"]) == k]
        b = [score(r, "test_r2_orig") for r in kz if r["arm"] == "random" and int(r["spec_id"]) == k]
        if a and b:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(set(a + b)) > 1 else 1.0
            print(f"  {label_of(k)} (extrapolates): full median {np.median(a):.4f}, random {np.median(b):.4f}, "
                  f"A12 {a12(a, b):.3f}, Mann-Whitney p={p:.3g}")
    print("\nDescriptive: by benchmark")
    for name, sel in (("Nguyen", lambda k: k < 100), ("Keijzer", lambda k: k > 100)):
        sub = [r for r in rows if sel(int(r["spec_id"]))]
        print_test(name, paired(sub, "full", "random"))


def analyse_noise(paths):
    for path in paths:
        rows = load(path)
        print_family(f"Noise run {Path(path).stem}", [paired(rows, "full", "random"), paired(rows, "full", "ssc")])
        describe(rows, ["random", "full", "ssc"])


def analyse_protocol(path):
    rows = load(path)
    print_family("Original Nguyen protocol (20 points, original ranges; 9 equations)", [
        paired(rows, "full", "random"),
        paired(rows, "full", "random", recovery_rate, "recovery rate"),
        paired(rows, "full", "ssc"),
    ])
    describe(rows, ["random", "full", "ssc"])


def analyse_circuits(path):
    rows = load(path)

    def per_target(arm, stat):
        by = defaultdict(list)
        for r in rows:
            if r["arm"] == arm:
                by[float(r["target"])].append(r)
        return {k: stat(v) for k, v in by.items()}

    def log_err(runs):  # lower is better, so negate: a positive difference favours the first arm
        return -float(np.median([math.log10(max(float(r["best_error"]), 1e-12)) for r in runs]))

    def solved(runs):
        return float(np.mean([r["solved"] == "1" for r in runs]))

    def test(a, b, stat, name):
        va, vb = per_target(a, stat), per_target(b, stat)
        d = np.array([va[k] - vb[k] for k in sorted(set(va) & set(vb))])
        p = float(wilcoxon(d, zero_method="wilcox").pvalue) if (d != 0).any() else 1.0
        return f"{a} vs {b} ({name})", int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), p

    print_test("Primary test (over targets; per-target median log10 relative error)",
               test("learned", "swap", log_err, "log10 error"))
    print_family("Secondary tests", [
        test("learned", "uniform", log_err, "log10 error"),
        test("swap", "uniform", log_err, "log10 error"),
        test("learned", "swap", solved, "solved fraction"),
    ])
    print(f"\n{'arm':8s} {'runs':>5s} {'med error':>10s} {'solved':>7s} {'resistors':>9s} {'evals':>7s} {'time s':>7s}")
    for arm in ("swap", "learned", "uniform"):
        rs = [r for r in rows if r["arm"] == arm]
        if rs:
            print(f"{arm:8s} {len(rs):5d} {np.median([float(r['best_error']) for r in rs]):10.2e} {solved(rs):7.3f} "
                  f"{np.median([int(r['best_resistors']) for r in rs]):9.0f} "
                  f"{np.mean([int(r['evaluations']) for r in rs]):7.0f} {np.mean([float(r['elapsed_sec']) for r in rs]):7.2f}")


def figures(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load(path)
    out = RESULTS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    eqs = sorted({int(r["spec_id"]) for r in rows})

    # Box plots of test R² per equation, full vs random
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (arm, color) in enumerate((("random", "#888888"), ("full", "#1A6EBD"))):
        data = [[score(r) for r in rows if r["arm"] == arm and int(r["spec_id"]) == k] for k in eqs]
        bp = ax.boxplot(data, positions=np.arange(len(eqs)) * 3 + i, widths=0.8, patch_artist=True, showfliers=True)
        for b in bp["boxes"]:
            b.set_facecolor(color)
        ax.plot([], [], color=color, lw=6, label=arm)
    ax.set_xticks(np.arange(len(eqs)) * 3 + 0.5, [label_of(k) for k in eqs])
    ax.set_ylabel("test R² (negative counted as 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "boxplots.pdf")

    # Convergence: median over runs of the best train R² against generation, evaluations and time
    curves = [json.loads(l) for l in open(Path(path).with_name(Path(path).stem + "_curves.jsonl"), encoding="utf-8")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm in ("random", "full", "ssc", EQUAL_COMPUTE):
        cs = [c["curve"] for c in curves if c["arm"] == arm and c["curve"]]
        if not cs:
            continue
        n = max(len(c) for c in cs)
        # Runs that stopped early (R² = 1) keep their last value
        r2 = np.array([[max(p[0], 0) for p in c] + [max(c[-1][0], 0)] * (n - len(c)) for c in cs])
        axes[0].plot(np.median(r2, axis=0), label=arm)
        for ax, j in ((axes[1], 2), (axes[2], 1)):
            grid = np.linspace(0, np.median([c[-1][j] for c in cs]), 200)
            vals = np.array([np.interp(grid, [p[j] for p in c], [max(p[0], 0) for p in c]) for c in cs])
            ax.plot(grid, np.median(vals, axis=0), label=arm)
    for ax, xl in zip(axes, ("generation", "fitness evaluations", "seconds")):
        ax.set_xlabel(xl)
        ax.set_ylabel("median best train R²")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "convergence.pdf")
    print(f"wrote {out / 'boxplots.pdf'} and {out / 'convergence.pdf'}")


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "main":
        analyse_main(args[0] if args else RESULTS / "main.csv")
    elif cmd == "noise":
        analyse_noise(args or [RESULTS / "noise01.csv", RESULTS / "noise10.csv"])
    elif cmd == "protocol":
        analyse_protocol(args[0] if args else RESULTS / "nguyen_orig.csv")
    elif cmd == "circuits":
        analyse_circuits(args[0] if args else RESULTS / "circuits.csv")
    elif cmd == "figures":
        figures(args[0] if args else RESULTS / "main.csv")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
