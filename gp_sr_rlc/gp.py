"""The genetic programming core: seeding, device, operator set, expression trees, variation,
fitness, selection, constant optimization and optional SymPy simplification.

The operator tables, VAR_NAMES and DEVICE are rebound at the start of each run, so other
modules read them as `gp.NAME` rather than importing the names.
"""
import copy
import multiprocessing as mp
import operator
import os
import random
import time
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import sympy as sp
import torch
from scipy.optimize import minimize


# ============================================================================
# Seeding
# ============================================================================

DEFAULT_SEED = 0


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Device
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def resolve_selector_device(use_learned_crossover: bool, selector_device: str | None = "auto") -> torch.device:
    requested = os.getenv("GP_SR_RLC_SELECTOR_DEVICE", selector_device or "auto")
    requested = str(requested).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto mode: standard behavior (use CUDA if available).
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Operator set and variables
# ============================================================================

# Protected operators to avoid NaNs/Infs
def pdiv(a, b):
    return a / (b + 1e-6)


def ppow(a, b):
    # Avoid hard crashes on invalid domains; NaNs are handled by fitness.
    with np.errstate(all="ignore"):
        return np.power(a, b)


def plog(a):
    return np.log(np.abs(a) + 1e-10)


def pexp(a):
    # clip to avoid overflow
    return np.exp(np.clip(a, -50, 50))


DEFAULT_UNARY_FUNCS = {
    'sin': np.sin,
    'cos': np.cos,
    'exp': pexp,
    'log': plog,
}

BASE_BINARY_FUNCS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': pdiv,
}

# Sympy mappings for conversion/simplification
DEFAULT_SYMPY_UNARY = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': lambda a: sp.log(sp.Abs(a)),  # match plog, which takes log(|x|)
}


def _sympy_safe_div(a, b):
    if b == 0:
        return a / sp.Float(1e-12)
    return a / b


BASE_SYMPY_BINARY = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _sympy_safe_div,
}

BINARY_FUNCS = dict(BASE_BINARY_FUNCS)
SYMPY_BINARY = dict(BASE_SYMPY_BINARY)

UNARY_FUNCS = dict(DEFAULT_UNARY_FUNCS)
SYMPY_UNARY = {name: DEFAULT_SYMPY_UNARY[name] for name in UNARY_FUNCS}
ALL_UNARY = list(UNARY_FUNCS.keys())

ALL_BINARY = list(BINARY_FUNCS.keys())

VAR_NAMES = ['x']
VAR_SYMBOLS = {name: sp.Symbol(name) for name in VAR_NAMES}


def get_selector_feature_dim(base_dim: int = 10) -> int:
    return base_dim + len(ALL_UNARY) + len(ALL_BINARY)


def set_variables(names: List[str]):
    unique = list(dict.fromkeys(names))
    if not unique:
        raise ValueError("var_names must contain at least one name.")
    if len(unique) != len(names):
        raise ValueError("var_names must be unique.")
    global VAR_NAMES, VAR_SYMBOLS
    VAR_NAMES = unique
    VAR_SYMBOLS = {name: sp.Symbol(name) for name in VAR_NAMES}


def configure_operators(unary_funcs=None, allow_pow=False):
    """Set the unary operators (a dict name -> (numpy_func, sympy_func); None for the
    defaults) and the binary operators (the four base ones, plus pow if allowed)."""
    global UNARY_FUNCS, SYMPY_UNARY, ALL_UNARY
    if unary_funcs is not None:
        if not isinstance(unary_funcs, dict):
            raise ValueError("unary_funcs must be a dict mapping names to (numpy_func, sympy_func) tuples")
        UNARY_FUNCS = {}
        SYMPY_UNARY = {}
        for name, funcs in unary_funcs.items():
            if isinstance(funcs, tuple) and len(funcs) == 2:
                UNARY_FUNCS[name] = funcs[0]
                SYMPY_UNARY[name] = funcs[1]
            else:
                raise ValueError(f"Each unary function must be a (numpy_func, sympy_func) tuple, got {funcs}")
        ALL_UNARY = list(UNARY_FUNCS.keys())
    else:
        UNARY_FUNCS = dict(DEFAULT_UNARY_FUNCS)
        SYMPY_UNARY = {name: DEFAULT_SYMPY_UNARY[name] for name in UNARY_FUNCS}
        ALL_UNARY = list(UNARY_FUNCS.keys())

    # Configure binary functions (optional pow support)
    global BINARY_FUNCS, SYMPY_BINARY, ALL_BINARY
    BINARY_FUNCS = dict(BASE_BINARY_FUNCS)
    SYMPY_BINARY = dict(BASE_SYMPY_BINARY)
    if allow_pow:
        BINARY_FUNCS["pow"] = ppow
        SYMPY_BINARY["pow"] = lambda a, b: a ** b
    ALL_BINARY = list(BINARY_FUNCS.keys())


# ============================================================================
# Expression trees
# ============================================================================

class Node:
    def evaluate(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    def size(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def all_nodes(self) -> List["Node"]:
        raise NotImplementedError

    def to_sympy(self):
        raise NotImplementedError


class Const(Node):
    def __init__(self, value: float):
        self.value = float(value)

    def evaluate(self, inputs):
        reference = next(iter(inputs.values()))
        return np.full_like(reference, self.value, dtype=float)

    def size(self):
        return 1

    def depth(self):
        return 0

    def to_string(self):
        return f"{self.value:.4g}"

    def all_nodes(self):
        return [self]

    def to_sympy(self):
        return sp.Float(self.value)


class Var(Node):
    def __init__(self, name: str = None):
        self.name = name or VAR_NAMES[0]

    def evaluate(self, inputs):
        return inputs[self.name]

    def size(self):
        return 1

    def depth(self):
        return 0

    def to_string(self):
        return self.name

    def all_nodes(self):
        return [self]

    def to_sympy(self):
        return VAR_SYMBOLS[self.name]


class Unary(Node):
    def __init__(self, func_name: str, child: Node):
        self.func_name = func_name
        self.child = child

    def evaluate(self, inputs):
        f = UNARY_FUNCS[self.func_name]
        return f(self.child.evaluate(inputs))

    def size(self):
        return 1 + self.child.size()

    def depth(self):
        return 1 + self.child.depth()

    def to_string(self):
        return f"{self.func_name}({self.child.to_string()})"

    def all_nodes(self):
        return [self] + self.child.all_nodes()

    def to_sympy(self):
        f = SYMPY_UNARY[self.func_name]
        return f(self.child.to_sympy())


class Binary(Node):
    def __init__(self, func_name: str, left: Node, right: Node):
        self.func_name = func_name
        self.left = left
        self.right = right

    def evaluate(self, inputs):
        f = BINARY_FUNCS[self.func_name]
        return f(self.left.evaluate(inputs), self.right.evaluate(inputs))

    def size(self):
        return 1 + self.left.size() + self.right.size()

    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def to_string(self):
        op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}[self.func_name]
        return f"({self.left.to_string()} {op} {self.right.to_string()})"

    def all_nodes(self):
        return [self] + self.left.all_nodes() + self.right.all_nodes()

    def to_sympy(self):
        f = SYMPY_BINARY[self.func_name]
        return f(self.left.to_sympy(), self.right.to_sympy())


# Helpers to pick random node and replace subtree
def collect_nodes_with_parents(node: Node, parent=None, attr_name=None):
    res = [(node, parent, attr_name)]
    if isinstance(node, Unary):
        res += collect_nodes_with_parents(node.child, node, 'child')
    elif isinstance(node, Binary):
        res += collect_nodes_with_parents(node.left, node, 'left')
        res += collect_nodes_with_parents(node.right, node, 'right')
    return res


def gather_used_var_names(ind: Node) -> set[str]:
    return {node.name for node in ind.all_nodes() if isinstance(node, Var)}


def replace_child(parent: Node, attr_name: str, new_child: Node):
    if parent is None:
        raise ValueError("replace_child called with parent=None")
    if isinstance(parent, Unary) and attr_name == 'child':
        parent.child = new_child
    elif isinstance(parent, Binary) and attr_name in ('left', 'right'):
        if attr_name == 'left':
            parent.left = new_child
        else:
            parent.right = new_child
    else:
        setattr(parent, attr_name, new_child)


# ============================================================================
# Inputs
# ============================================================================

def prepare_inputs(x, var_names=None) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if isinstance(x, Mapping):
        data = {key: np.asarray(val, dtype=float).reshape(-1) for key, val in x.items()}
        inferred = list(data.keys())
        if var_names is None:
            var_names = inferred
        else:
            missing = set(var_names) - data.keys()
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Missing inputs for variables: {missing_list}")
            data = {name: data[name] for name in var_names}
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            if var_names is None:
                var_names = ['x']
            elif len(var_names) != 1:
                raise ValueError("var_names length must match x shape.")
            data = {var_names[0]: arr}
        elif arr.ndim == 2:
            n_vars = arr.shape[1]
            if var_names is None:
                var_names = [f'x{i}' for i in range(n_vars)]
            elif len(var_names) != n_vars:
                raise ValueError("var_names length must match number of columns in x.")
            data = {name: arr[:, idx] for idx, name in enumerate(var_names)}
        else:
            raise ValueError("x must be 1D, 2D, or a mapping of arrays.")
    lengths = {len(col) for col in data.values()}
    if len(lengths) != 1:
        raise ValueError("All variable arrays must share the same length.")
    return data, list(var_names)


# ============================================================================
# Random trees and mutation
# ============================================================================

# Random tree generation with configurable constants
def random_tree(max_depth: int, prob_const=0.3, const_range=(-5, 5)) -> Node:
    if max_depth == 0:
        if random.random() < prob_const:
            return Const(random.uniform(*const_range))
        return Var(random.choice(VAR_NAMES))

    # Build choices dynamically so empty operator sets don't crash.
    choices = ["const", "var"]
    weights = [0.2, 0.2]
    if ALL_UNARY:
        choices.append("unary")
        weights.append(0.3)
    if ALL_BINARY:
        choices.append("binary")
        weights.append(0.3)

    total = sum(weights)
    pick = random.random() * total
    cumulative = 0.0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if pick <= cumulative:
            if choice == "const":
                return Const(random.uniform(*const_range))
            if choice == "var":
                return Var(random.choice(VAR_NAMES))
            if choice == "unary":
                func = random.choice(ALL_UNARY)
                return Unary(func, random_tree(max_depth - 1, prob_const, const_range))
            func = random.choice(ALL_BINARY)
            return Binary(
                func,
                random_tree(max_depth - 1, prob_const, const_range),
                random_tree(max_depth - 1, prob_const, const_range),
            )

    # Fallback (should not hit)
    return Var()


def subtree_crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    a_clone = a.clone()
    b_clone = b.clone()
    a_nodes = collect_nodes_with_parents(a_clone)
    b_nodes = collect_nodes_with_parents(b_clone)
    # avoid replacing the entire tree in both directions maybe
    node_a, parent_a, attr_a = random.choice(a_nodes)
    node_b, parent_b, attr_b = random.choice(b_nodes)
    # Swap node_a and node_b in clones
    if parent_a is None:
        new_a = node_b.clone()
    else:
        new_a = a_clone
        replace_child(parent_a, attr_a, node_b.clone())
    if parent_b is None:
        new_b = node_a.clone()
    else:
        new_b = b_clone
        replace_child(parent_b, attr_b, node_a.clone())
    return new_a, new_b


def point_mutation(ind: Node, prob_change=0.1) -> Node:
    ind_clone = ind.clone()
    for node, parent, attr in collect_nodes_with_parents(ind_clone):
        if isinstance(node, Const):
            if random.random() < prob_change:
                node.value += random.uniform(-1, 1)
        elif isinstance(node, Unary):
            if random.random() < prob_change:
                node.func_name = random.choice(ALL_UNARY)
        elif isinstance(node, Binary):
            if random.random() < prob_change:
                node.func_name = random.choice(ALL_BINARY)
        elif isinstance(node, Var) and len(VAR_NAMES) > 1:
            if random.random() < prob_change:
                node.name = random.choice(VAR_NAMES)
    return ind_clone


def subtree_mutation(ind: Node, max_depth: int, prob_const=0.3, const_range=(-5, 5)) -> Node:
    ind_clone = ind.clone()
    nodes = collect_nodes_with_parents(ind_clone)
    node, parent, attr = random.choice(nodes)
    new_sub = random_tree(max_depth=random.randint(0, max_depth), prob_const=prob_const, const_range=const_range)
    if parent is None:
        return new_sub
    else:
        replace_child(parent, attr, new_sub)
        return ind_clone


def too_big(ind: Node, max_depth: int) -> bool:
    return ind.depth() > max_depth


# ============================================================================
# Fitness
# ============================================================================

def fitness(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
            complexity_coef=1e-3, inclusion_coef=1.0) -> float:
    try:
        y_pred = ind.evaluate(inputs)
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            return 1e6 + ind.size()
        mse = float(np.mean((y - y_pred)**2))
    except Exception:
        return 1e6 + ind.size()
    inclusion_penalty = 0.0
    if inclusion_coef:
        used_vars = gather_used_var_names(ind)
        missing_vars = set(inputs.keys()) - used_vars
        missing_vars |= detect_inert_variables(ind, inputs, y_pred, candidate_vars=used_vars)
        inclusion_penalty = inclusion_coef * len(missing_vars)
    return mse + complexity_coef * ind.size() + inclusion_penalty


def pareto_objectives(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
                      inclusion_coef=1.0) -> Tuple[float, float]:
    """Return (error_with_inclusion, complexity) objectives for Pareto sorting."""
    try:
        y_pred = ind.evaluate(inputs)
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            return 1e6 + ind.size(), float(ind.size())
        mse = float(np.mean((y - y_pred) ** 2))
    except Exception:
        return 1e6 + ind.size(), float(ind.size())
    inclusion_penalty = 0.0
    if inclusion_coef:
        used_vars = gather_used_var_names(ind)
        missing_vars = set(inputs.keys()) - used_vars
        missing_vars |= detect_inert_variables(ind, inputs, y_pred, candidate_vars=used_vars)
        inclusion_penalty = inclusion_coef * len(missing_vars)
    return mse + inclusion_penalty, float(ind.size())


def detect_inert_variables(
    ind: Node,
    inputs: Mapping[str, np.ndarray],
    base_pred: np.ndarray,
    atol=1e-5,
    rtol=1e-4,
    candidate_vars: set[str] | None = None,
) -> set[str]:
    """Variables that appear in the tree but don't change its output when shuffled."""
    if candidate_vars is None:
        var_names = [name for name in inputs.keys()]
    else:
        var_names = [name for name in candidate_vars if name in inputs]

    inert = set()
    if not var_names:
        return inert

    perturbed = dict(inputs)
    rolled_inputs = {name: np.roll(inputs[name], 1) for name in var_names}

    for name in var_names:
        original = inputs[name]
        perturbed[name] = rolled_inputs[name]
        try:
            alt_pred = ind.evaluate(perturbed)
        except Exception:
            inert.add(name)
            perturbed[name] = original
            continue
        if np.allclose(base_pred, alt_pred, atol=atol, rtol=rtol):
            inert.add(name)
        perturbed[name] = original
    return inert


# ============================================================================
# Selection
# ============================================================================

def tournament_select(pop: List[Node], scores: List[float], k=3, timeout=10.0) -> Node:
    if not pop:
        raise ValueError("Population must contain at least one individual.")
    best = tournament_pick_index(
        pop_size=len(pop),
        score_key=lambda i: scores[i],
        k=k,
        timeout=timeout,
    )
    return pop[best].clone()


def tournament_pick_index(pop_size: int, score_key, k=3, timeout=10.0) -> int:
    if pop_size <= 0:
        raise ValueError("Population size must be > 0.")
    deadline = time.perf_counter() + timeout
    sample_size = min(k, pop_size)
    while True:
        try:
            selected_idx = random.sample(range(pop_size), sample_size)
            return min(selected_idx, key=score_key)
        except Exception:
            if time.perf_counter() >= deadline:
                return random.randrange(pop_size)


def _pareto_dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_rank_and_crowding(objectives: List[Tuple[float, ...]]):
    """Compute Pareto rank and crowding distance (NSGA-II style)."""
    n = len(objectives)
    if n == 0:
        return [], []
    num_obj = len(objectives[0]) if objectives[0] else 0
    if num_obj == 0:
        return [0] * n, [0.0] * n
    domination_sets = [set() for _ in range(n)]
    domination_counts = [0] * n
    fronts = []

    for i in range(n):
        for j in range(i + 1, n):
            if _pareto_dominates(objectives[i], objectives[j]):
                domination_sets[i].add(j)
                domination_counts[j] += 1
            elif _pareto_dominates(objectives[j], objectives[i]):
                domination_sets[j].add(i)
                domination_counts[i] += 1

    current_front = [i for i in range(n) if domination_counts[i] == 0]
    rank = [0] * n
    front_rank = 0
    while current_front:
        fronts.append(current_front)
        for idx in current_front:
            rank[idx] = front_rank
            for dominated in domination_sets[idx]:
                domination_counts[dominated] -= 1
        next_front = []
        for idx in current_front:
            for dominated in domination_sets[idx]:
                if domination_counts[dominated] == 0:
                    next_front.append(dominated)
        # Deduplicate next front
        current_front = list(dict.fromkeys(next_front))
        front_rank += 1

    crowding = [0.0] * n
    for front in fronts:
        if len(front) <= 2:
            for idx in front:
                crowding[idx] = float("inf")
            continue
        for obj_idx in range(num_obj):
            sorted_front = sorted(front, key=lambda i: objectives[i][obj_idx])
            min_val = objectives[sorted_front[0]][obj_idx]
            max_val = objectives[sorted_front[-1]][obj_idx]
            crowding[sorted_front[0]] = float("inf")
            crowding[sorted_front[-1]] = float("inf")
            if max_val == min_val:
                continue
            for i in range(1, len(sorted_front) - 1):
                prev_val = objectives[sorted_front[i - 1]][obj_idx]
                next_val = objectives[sorted_front[i + 1]][obj_idx]
                crowding[sorted_front[i]] += (next_val - prev_val) / (max_val - min_val)
    return rank, crowding


def tournament_select_pareto(pop: List[Any], ranks: List[int], crowding: List[float], scores: List[float],
                             k=3, timeout=10.0) -> Any:
    if not pop:
        raise ValueError("Population must contain at least one individual.")
    best = tournament_pick_index(
        pop_size=len(pop),
        score_key=lambda i: (ranks[i], -crowding[i], scores[i]),
        k=k,
        timeout=timeout,
    )
    return pop[best].clone()


# ============================================================================
# Constant optimization
# ============================================================================

def optimize_constants(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
                       method='L-BFGS-B', maxiter=50) -> Node:
    """Optimize constants in the tree using numerical optimization while keeping structure fixed."""
    # Collect all constant nodes
    const_nodes = [node for node in ind.all_nodes() if isinstance(node, Const)]

    if not const_nodes:
        return ind  # No constants to optimize

    # Extract initial values
    initial_values = np.array([node.value for node in const_nodes], dtype=float)

    # Define objective function (MSE)
    def objective(const_values):
        # Temporarily update constant values
        for node, val in zip(const_nodes, const_values):
            node.value = float(val)

        try:
            y_pred = ind.evaluate(inputs)
            if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
                return 1e10  # Large penalty for invalid predictions
            mse = float(np.mean((y - y_pred)**2))
            return mse
        except Exception:
            return 1e10

    # Optimize
    try:
        result = minimize(
            objective,
            initial_values,
            method=method,
            options={'maxiter': maxiter}
        )

        # Update constants with optimized values
        if result.success:
            for node, val in zip(const_nodes, result.x):
                node.value = float(val)
        else:
            for node, val in zip(const_nodes, initial_values):
                node.value = float(val)
    except Exception:
        for node, val in zip(const_nodes, initial_values):
            node.value = float(val)

    return ind


# ============================================================================
# SymPy simplification (simplify_new_genomes)
# ============================================================================

def _timed_simplify_worker(expr, queue):
    try:
        queue.put(sp.simplify(expr))
    except Exception as exc:
        queue.put(exc)


def timed_sympy_simplify(expr: sp.Expr, timeout: float | None):
    if timeout is None or timeout <= 0:
        return None
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_timed_simplify_worker, args=(expr, queue))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    if queue.empty():
        return None
    result = queue.get()
    if isinstance(result, Exception):
        return None
    return result


def sympy_to_node(expr: sp.Expr) -> Node:
    if expr.is_Number:
        return Const(float(expr))
    if expr.is_Symbol:
        name = str(expr)
        if name not in VAR_SYMBOLS:
            raise ValueError(f"Unknown symbol '{name}' in simplified expression.")
        return Var(name)
    sympy_func_to_name = {func: name for name, func in SYMPY_UNARY.items()}
    if expr.func in sympy_func_to_name:
        if len(expr.args) != 1:
            raise ValueError("Unsupported arity for unary function.")
        return Unary(sympy_func_to_name[expr.func], sympy_to_node(expr.args[0]))
    if expr.func == sp.Add:
        args = list(expr.args)
        node = sympy_to_node(args[0])
        for arg in args[1:]:
            node = Binary('add', node, sympy_to_node(arg))
        return node
    if expr.func == sp.Mul:
        numerators, denominators = [], []
        for arg in expr.args:
            if isinstance(arg, sp.Pow) and arg.exp == -1:
                denominators.append(arg.base)
            else:
                numerators.append(arg)
        node = sympy_to_node(numerators[0]) if numerators else Const(1.0)
        for arg in numerators[1:]:
            node = Binary('mul', node, sympy_to_node(arg))
        for arg in denominators:
            node = Binary('div', node, sympy_to_node(arg))
        return node
    if isinstance(expr, sp.Pow):
        if 'pow' in ALL_BINARY:
            return Binary('pow', sympy_to_node(expr.base), sympy_to_node(expr.exp))
        if expr.exp == -1:
            return Binary('div', Const(1.0), sympy_to_node(expr.base))
        if expr.exp == 2:
            base_node = sympy_to_node(expr.base)
            return Binary('mul', base_node.clone(), base_node.clone())
        raise ValueError("Unsupported exponent encountered during simplification.")
    raise ValueError("Unsupported sympy construct during simplification.")


def simplify_node_with_timeout(node: Node, timeout: float | None) -> Node:
    if timeout is None or timeout <= 0:
        return node
    try:
        simplified_expr = timed_sympy_simplify(node.to_sympy(), timeout)
    except Exception:
        return node
    if simplified_expr is None:
        return node
    try:
        return sympy_to_node(simplified_expr)
    except ValueError:
        return node
