# Revision experiments: plan

**Status: approved by the author on 2026-11-15. No planned run has started.** Once approved, this file is committed
and tagged `plan-frozen`; nothing in it, in `experiments.py` or in `analysis.py` changes after the first run
starts, and the results are reported whichever way they go. Any departure goes in `plans/DEVIATIONS.md`.

## Method under test
GP-SR-RLC as published (rllxo 0da8b7f): the per-node MLP policy plus the structural heuristic, the joint
pair softmax, the A2C update, trained from scratch in every run. Settings are the paper's
(`benchmarks.ga_config` + `learned_overrides`: population 300, 200 generations, tournament 9, 24 elites,
mutation 0.5, max depth 30, Pareto selection, constant optimization every 10 generations, warmup 30,
temperature 1.0, heuristic scale 0.08). Nothing is tuned.

Code changes since the published version, all bug fixes or options that are off by default:
- invalid trees score 1e18 + size instead of 1e6 + size, and `run_ga` returns the final tree's own R²;
- resistor pruning drops every resistor that carries no current;
- three comparison operators (`heuristic`, `ssc`, `sizefair`) and a like-for-like `swap` resistor
  crossover, used only by the arms that name them;
- linear-terms mode (climate only) is gone, along with three random draws per child it made even when
  off, so a seed no longer reproduces a published run;
- speed only: identical trees are scored once and the score reused, and node features and logits are
  cached per tree. On 72 full-length dev runs every result is identical to the code without these
  changes, and the runs take 2.4x less time.

## Decisions (approved 2026-11-15)
1. **Function sets: each benchmark's standard set** for every equation (Nguyen {+, -, *, /, sin, cos,
   exp, log}, Uy et al. 2011; Keijzer the same plus sqrt; no pow). The published runs gave each equation
   only the functions in its true answer, plus pow for N11 and K10, which hands the search part of the
   answer. Reviewer 5 asked for the standard protocol.
2. **Equations: all 27 Nguyen and Keijzer equations** (N1-12, K1-15). The first revision held out N6, N7,
   N12 and K14 because it developed the now-dropped semantic-bias method on them. The method tested here
   uses the published settings and was tuned on none of them; in this revision they served only for
   smoke tests and the timing pilot.
3. **Equal compute: `random_x3`**, random crossover with 600 generations. In the timing pilot (dev
   equations, 2 seeds, 10 workers) a full-method run made 2.2x the fitness evaluations of a random run
   (133,000 vs 60,000, counting the selector's reward evaluations), so 3x the generations gives random
   crossover at least the method's evaluations. Equal generations stays the primary comparison; wall
   time is reported for every arm but not matched, since it depends on the implementation.
4. **Machine: the laptop**, 10 workers. A worker holds about 0.5 GB of its own memory. Runs may be split
   over several sessions; the runner skips finished runs.

## Data and scoring (all SR runs)
- Training: 512 points from the ranges in `benchmarks.py`, drawn with seed `seed + 1000 * spec_id`
  (the published runner's sample). Test: 512 fresh points from the same ranges (seed `10000 + ...`).
- Each run records train and test R², exact recovery, Zhang-Shasha tree edit distance (TED) and NED to the
  true equation, model size, wall time, fitness evaluations (one per individual per generation, and
  separately the selector's reward evaluations), and per-generation (best train R², seconds, evaluations).
  Evaluations are counted per request, whether or not the score came from the cache, so the count does
  not depend on the caching. Constant optimization and SSC's subtree evaluations are not counted.
- Keijzer runs also score the final model on Keijzer's (2003) test sets (`test_r2_orig`); K4 and K6 extrapolate.
- **Recovery**: model minus true equation simplifies to 0 after snapping constants (to the target's own
  float constants within 0.1%, to small rationals times powers of e or pi within 0.01%, else to the nearest
  simple rational within 1e-3), with real variables. Each SymPy simplification stops after 30 s or on
  an error (recovery: not recovered; TED: computed on the unsimplified form). The check runs in a helper
  process killed after 60 s (counts as not recovered); TED after 150 s (NaN).
- **Failed runs**: every worker and its helper are capped at 3 GB of memory, so a run that grows past
  that fails instead of swapping. A run that ends in an error is rerun once with the same command (the runner skips
  finished runs). If it fails again it counts as 0 and is listed in `plans/DEVIATIONS.md`.
- **Score of a run**: test R², with negative, non-finite or error counted as 0.
- **Score of an arm on an equation**: the median over its 30 seeds (0-29).
- **Paired tests** are two-sided Wilcoxon signed-rank over equations, zero differences dropped,
  alpha = 0.05. With so few equations (27, or 12 and 15 per benchmark), a non-significant result is reported as inconclusive.

## Run 1: main comparison and ablations (R4.1-R4.5, R5.1-R5.5)
| Arm | What it is |
|---|---|
| random | uniform crossover points (the baseline) |
| full | GP-SR-RLC: policy plus heuristic |
| policy | policy alone (heuristic scale 0); also the independent-sampling arm, since without the heuristic the pair softmax factorizes |
| heuristic | heuristic alone, no network: the method's pair distribution with zero node logits |
| nocritic | full without the critic |
| full_noconst, random_noconst | without constant optimization |
| full_m01, random_m01 | mutation rate 0.1 instead of 0.5, where crossover dominates the search |
| ssc | semantic similarity-based crossover (Uy et al., 2011: bounds 1e-4 and 0.4, 12 trials) |
| sizefair | size-fair crossover (Langdon, 2000) |
| random_x3 | random crossover with 3x the generations (equal compute) |

```
python experiments.py sr --equations all --arms random,full,policy,heuristic,nocritic,full_noconst,random_noconst,full_m01,random_m01,ssc,sizefair,random_x3 --seeds 30 --jobs 10 --tag main
python analysis.py main
python analysis.py figures
```
9,720 runs, about 10.5 hours on the laptop (from the pilot's run times, including the recovery checks).

**Primary test:** full vs random on per-equation median test R².

**Secondary tests, one Holm-corrected family:**
- full vs random on solved fraction (test R² >= 0.999) and on recovery rate;
- full vs heuristic, full vs policy, policy vs random, heuristic vs random (what the learning adds);
- full vs nocritic;
- full_noconst vs random_noconst, full_m01 vs random_m01;
- full vs ssc, full vs sizefair, ssc vs random, sizefair vs random;
- full vs random_x3.

**Descriptive:** per arm, mean and median test R², solved fraction, recovery rate, median size, time and
evaluations; per equation, full vs random by Mann-Whitney U (Holm over equations) with the Vargha-Delaney
A12; Keijzer's original test sets (paired test over the 15 equations, K4 and K6 separately); Nguyen and Keijzer
separately; box plots per equation; convergence against generations, evaluations and time.

## Run 2: noise (R5.2)
Gaussian noise on the training targets only, sd = level x RMS(y) as in SRBench; the test set stays clean.
```
python experiments.py sr --equations all --arms random,full,ssc --seeds 30 --jobs 10 --noise 0.01 --tag noise01
python experiments.py sr --equations all --arms random,full,ssc --seeds 30 --jobs 10 --noise 0.1 --tag noise10
python analysis.py noise
```
4,860 runs, about 4.5 hours. **Per level, one Holm family:** full vs random, full vs ssc.

## Run 3: the original Nguyen protocol (R4.6, R5.2)
The 12 Nguyen equations trained on 20 points from Uy et al.'s ranges, tested on 512 fresh points
from the same ranges.
```
python experiments.py sr --equations nguyen --arms random,full,ssc --seeds 30 --jobs 10 --protocol orig --tag nguyen_orig
python analysis.py protocol
```
1,080 runs, about 1 hour. **One Holm family:** full vs random (test R²), full vs random (recovery rate),
full vs ssc.

## Run 4: resistor networks (R2.4, R5.6)
6 nodes, 6 resistors, E6 values from 10 ohm to 6.8 kohm; population 100, 50 generations, the paper's
resistor GA otherwise. 30 targets log-uniform between 5 ohm and 5 kohm (seed 123), seeds 0-9.

| Arm | Crossover |
|---|---|
| swap | swap one uniformly random resistor pair (the like-for-like baseline) |
| learned | the paper's selector picks the pair to swap |
| uniform | the paper's uniform crossover (each position swapped with probability 0.5) |

```
python experiments.py circuits --arms swap,learned,uniform --targets 30 --seeds 10 --jobs 10
python analysis.py circuits
```
900 runs, a few minutes. The outcome of a run is the smallest relative error on its final Pareto front;
"solved" is below 1e-4. **Primary:** learned vs swap, Wilcoxon over the 30 targets of the per-target median
log10 error. **Secondary, one Holm family:** learned vs uniform, swap vs uniform, learned vs swap on solved
fraction. Resistor counts and evaluations (distinct networks scored) are reported.

## Not in this plan
- **PySR context table (R4.1, R5.1).** It needs a separate PySR/Julia environment and its own scripts; it
  will get its own section here, approved before it runs. It is context only, not a test.
- Reviewer requests answered in the text without new runs: NeuroCrossover and Deep Neural Crossover
  (argued, not implemented), gplearn/Operon/DSR head-to-head (argued), the RL formulation questions (R5.5:
  trained from scratch per run, a contextual bandit, reward standardization).
