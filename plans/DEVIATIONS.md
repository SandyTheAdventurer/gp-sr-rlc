# Deviations from the frozen plan (tag `plan-frozen`)

## 1. Two noise runs rerun after a timeout in the recovery check (2026-11-16)
Of the 18,990 planned runs, two in Run 2 at noise 0.1 ended with status error, `TimeoutError()`:
random on K9 seed 5, and full on N11 seed 15. In both, the GA had finished (test R² 0.9974 and 0.9995)
and the error came from the recovery check afterwards.

Cause (my reading, not reproduced): each SymPy simplification runs under a 30 s `SIGALRM`. If SymPy is
inside one long C call when the alarm fires, Python raises the `TimeoutError` only after that call
returns. That can happen inside the `finally` that turns the alarm off, outside the `try` meant to catch it,
so the exception escapes as a run error. Checked directly, outside the helper, both models finish
`is_recovered` and `tree_edit_distance` in under 5 s, so the failure depends on timing.

As the plan says, each was rerun once with the same command and unchanged code. Because the runner counts
error rows as finished, the two rows (and their curves) were first removed from `results/noise10.csv` and
`results/noise10_curves.jsonl`; the originals are kept as `results/noise10.before_rerun.*`. Both reruns
finished with status ok and the same test R²: random K9 seed 5 not recovered, TED 18; full N11 seed 15 not
recovered, TED 14. Their `elapsed_sec` was measured with 2 runs in parallel, not 10.

The code was not changed: every run had already finished, and the race can only cause an error, not a wrong
value. All 18,990 runs end with status ok.

## 2. A stale label in the protocol analysis (2026-11-16)
`analysis.py protocol` printed "9 equations" in its title, left over from the held-out set. The title now
counts the equations in the data (12). Nothing else in the analysis changed.

## 3. Run 5: runs that stopped before the warmup (2026-11-16)
52 of the 540 `_diag` runs found an exact solution before generation 30, when the policy starts acting, so
their diagnostic files are empty: every seed of K7 and K8 in both arms, and 6 other runs per arm. The `diag`
analysis crashed on the first empty file. It now skips these runs and lists them. The diagnostic tests
therefore cover 25 equations (K7 and K8 have no diagnostics). The tests on test R² are unaffected.
