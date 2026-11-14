# GP-SR-RLC

Genetic programming for symbolic regression with a crossover operator learned by
reinforcement learning. An actor-critic (A2C) network scores the pairs of crossover points
in two parent trees, combined with a structural heuristic, and is trained on whether the
offspring improve on their parents.

## Layout

| File | Contents |
| --- | --- |
| `gp_sr_rlc/gp.py` | the GP core: trees, operator set, mutation, fitness, selection, constant optimization |
| `gp_sr_rlc/crossover.py` | the actor-critic selector and the learned crossover with its A2C update |
| `gp_sr_rlc/ga.py` | `run_ga`, the main loop |
| `benchmarks.py` | Nguyen and Keijzer equations, one run per (equation, seed, mode), edit distance |
| `run_benchmark.py` | learned vs random crossover on the benchmarks, with reports |
| `circuits.py` | the resistor-network application |
| `run_circuit.py` | evolve a resistor network to a target resistance, and draw it |

## Running

```
pip install -r requirements.txt
python run_benchmark.py --benchmark all --jobs 8
python run_circuit.py --learned --target 53.9
```
