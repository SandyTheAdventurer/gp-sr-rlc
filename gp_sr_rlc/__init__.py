"""GP-SR-RLC: genetic programming for symbolic regression with a reinforcement-learned crossover.

    gp.py         the GP core: trees, operator set, mutation, fitness, selection, constants
    crossover.py  the actor-critic selector and the learned crossover with its A2C update
    ga.py         run_ga, the main loop
"""
import warnings

from .gp import DEFAULT_SEED, set_global_seed

warnings.filterwarnings("ignore", category=RuntimeWarning)
set_global_seed(DEFAULT_SEED)

from .crossover import SubtreeSelector  # noqa: E402
from .ga import run_ga  # noqa: E402
from .gp import DEVICE, Binary, Const, Node, Unary, Var  # noqa: E402  (DEVICE: the device at import time)

__all__ = [
    "DEFAULT_SEED", "DEVICE", "set_global_seed", "run_ga", "SubtreeSelector",
    "Node", "Const", "Var", "Unary", "Binary",
]
