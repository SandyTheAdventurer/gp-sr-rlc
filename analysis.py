"""Analysis of the revision's runs, exactly as plans/REVISION_PLAN.md specifies.

Written before any of the planned runs. Each command reads one results CSV and prints its tests;
`figures` writes the box plots and convergence curves to results/figs/.

Usage:
    python analysis.py main      [results/main.csv]
    python analysis.py noise     results/noise01.csv results/noise10.csv
    python analysis.py protocol  [results/nguyen_orig.csv]
    python analysis.py circuits  [results/circuits.csv]
    python analysis.py figures   [results/main.csv]
    python analysis.py diag      (results/diag.csv, results/oracle.csv, results/main.csv, results/diag/diag/)

`diag` (Run 5) was written after Runs 1-4 had been analysed, and before any Run 5 run.
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
# Random crossover with 3x the generations: about the full method's evaluations per run (2.2x random in
# the dev-equation timing pilot, counting the selector's reward evaluations). Fixed in the plan.
EQUAL_COMPUTE = ("random_x3",)

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
            "random_m01", "ssc", "sizefair", *EQUAL_COMPUTE]
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
        *[paired(rows, "full", arm) for arm in EQUAL_COMPUTE],
    ])
    print("\nDescriptive")
    describe(rows, arms)
    per_equation_tests(rows, "full", "random")
    kz = [r for r in rows if int(r["spec_id"]) > 100]
    print_test(f"Descriptive: Keijzer's original test sets ({len({r['spec_id'] for r in kz})} equations)",
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
    print_family(f"Original Nguyen protocol (20 points, original ranges; {len({r['spec_id'] for r in rows})} equations)", [
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
    for arm in ("random", "full", "ssc", *EQUAL_COMPUTE):
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


# ============================================================================
# Run 5: why the learned crossover matches random
# ============================================================================


def diag_runs(directory):
    """{(arm, spec_id, seed): per-run means of the per-generation diagnostics}."""
    out, empty = {}, []
    for f in sorted(Path(directory).glob("*.json")):
        arm, spec_id, seed = f.stem.rsplit("_", 2)
        d = json.load(open(f, encoding="utf-8"))
        gens = sorted(d, key=int)
        if not gens:  # stopped (solved) before the warmup ended: the policy never acted
            empty.append((arm, int(spec_id), int(seed)))
            continue
        g = [d[k] for k in gens]
        hr = [x["headroom"] for x in g if x["headroom"]]
        last = int(gens[-1])
        m = lambda key, sel=g: float(np.mean([x[key] for x in sel]))
        out[(arm, int(spec_id), int(seed))] = {
            "kl": m("kl"), "hn": m("hn"), "max_ratio": m("max_ratio"), "drawn_ratio": m("drawn_ratio"),
            "kl_net": m("kl_net"), "kl_heur": m("kl_heur"), "n": m("n"),
            "kl_early": m("kl", [x for k, x in zip(gens, g) if int(k) < 60]),
            "kl_late": m("kl", [x for k, x in zip(gens, g) if int(k) > last - 30]),
            "lift": float(np.mean([h["frac_better_p"] - h["frac_better"] for h in hr])),
            "drawn_pct": float(np.mean([h["drawn_pct"] for h in hr])),
            "rho": float(np.nanmedian([h["rho"] for h in hr])) if any(np.isfinite(h["rho"]) for h in hr) else float("nan"),
            "frac_better": float(np.mean([h["frac_better"] for h in hr])),
            "p_best16": float(np.mean([1 - (1 - h["frac_better"]) ** 16 for h in hr])),
            "r_gain": float(np.mean([h["r_max"] > 0 for h in hr])),
        }
    if empty:
        print(f"{len(empty)} runs stopped before the warmup ended and have no diagnostics: "
              + ", ".join(f"{a} eq={k} seed={sd}" for a, k, sd in empty))
    return out


def diag_per_equation(runs, arm, key):
    by = defaultdict(list)
    for (a, k, _), v in runs.items():
        if a == arm and np.isfinite(v[key]):
            by[k].append(v[key])
    return {k: float(np.median(v)) for k, v in by.items()}


def diag_paired(runs, a, b, key):
    va, vb = diag_per_equation(runs, a, key), diag_per_equation(runs, b, key)
    d = np.array([va[k] - vb[k] for k in sorted(set(va) & set(vb))])
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    p = float(wilcoxon(d, zero_method="wilcox").pvalue) if (d != 0).any() else 1.0
    return f"{a} vs {b} ({key})", w, l, t, p


def diag_one_sample(runs, arm, key):
    d = np.array(list(diag_per_equation(runs, arm, key).values()))
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    p = float(wilcoxon(d, zero_method="wilcox").pvalue) if (d != 0).any() else 1.0
    return f"{arm}: {key} vs 0", w, l, t, p


def analyse_diag(diag_csv, oracle_csv, main_csv, diag_dir):
    rows_diag, rows_oracle, rows_main = load(diag_csv), load(oracle_csv), load(main_csv)
    runs = diag_runs(diag_dir)

    # The logging must not have changed the runs: full_diag reproduces main's full run for run
    main_full = {(r["spec_id"], r["seed"]): r for r in rows_main if r["arm"] == "full"}
    fd = [r for r in rows_diag if r["arm"] == "full_diag"]
    same = sum(main_full[(r["spec_id"], r["seed"])]["best_expr"] == r["best_expr"] for r in fd)
    print(f"Check: {same} of {len(fd)} full_diag runs reproduce the main run's full arm exactly")

    print("\nThe policy's pair distribution (per run: mean over generations after the warmup; "
          "table: median [10th, 90th percentile] over runs)")
    keys = [("kl", "KL(p || uniform), nats"), ("kl_early", "  first 30 generations"), ("kl_late", "  last 30 generations"),
            ("hn", "H(p) / log N"), ("max_ratio", "N * max p"), ("drawn_ratio", "N * p(drawn pair)"),
            ("kl_net", "KL, network alone"), ("kl_heur", "KL, heuristic alone"), ("n", "pairs N")]
    print(f"  {'':28s} {'full_diag':>26s} {'noent_diag':>26s}")
    for key, label in keys:
        cells = []
        for arm in ("full_diag", "noent_diag"):
            v = [x[key] for (a, _, _), x in runs.items() if a == arm]
            cells.append(f"{np.median(v):9.4f} [{np.percentile(v, 10):6.3f}, {np.percentile(v, 90):6.3f}]" if v else "-")
        print(f"  {label:28s} {cells[0]:>26s} {cells[1]:>26s}")
    kl = np.median([x["kl"] for (a, _, _), x in runs.items() if a == "full_diag"])
    kl_ne = np.median([x["kl"] for (a, _, _), x in runs.items() if a == "noent_diag"])
    print(f"  H1 (near-uniform: median run KL < 0.05 nats): {'supported' if kl < 0.05 else 'not supported'} "
          f"(median {kl:.4f})")

    print("\nHow much the point choice can matter (once per generation, every point pair scored; "
          "median [10th, 90th percentile] over runs)")
    for key, label in (("frac_better", "fraction of pairs whose better child beats the better parent"),
                       ("p_best16", "chance the best of 16 uniform pairs does"),
                       ("r_gain", "chance at least one pair does"),
                       ("lift", "policy minus uniform, fraction beating the better parent"),
                       ("drawn_pct", "drawn pair's reward percentile (0.5 = uniform)"),
                       ("rho", "Spearman(log p, reward)")):
        for arm in ("full_diag", "noent_diag"):
            v = [x[key] for (a, _, _), x in runs.items() if a == arm and np.isfinite(x[key])]
            if v:
                print(f"  {label:62s} {arm:10s} {np.median(v):8.4f} [{np.percentile(v, 10):7.4f}, {np.percentile(v, 90):7.4f}]")

    rows = rows_main + rows_oracle
    print_family("Run 5 tests", [
        diag_paired(runs, "noent_diag", "full_diag", "kl"),
        diag_one_sample(runs, "full_diag", "lift"),
        paired(rows, "best16", "random"),
        paired(rows, "worst16", "random"),
        paired(rows_diag, "noent_diag", "full_diag"),
    ])
    print("\nDescriptive")
    describe(rows_main + rows_oracle + rows_diag, ["random", "full", "best16", "worst16", "full_diag", "noent_diag"])


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
    elif cmd == "diag":
        analyse_diag(RESULTS / "diag.csv", RESULTS / "oracle.csv", RESULTS / "main.csv", RESULTS / "diag" / "diag")
    elif cmd == "figures":
        figures(args[0] if args else RESULTS / "main.csv")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
