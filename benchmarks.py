"""Nguyen and Keijzer benchmarks: the equations, one GA run per (equation, seed, mode), and
the edit distance between the evolved tree and the target.
"""
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sympy as sp

from gp_sr_rlc import Binary, Const, Node, Unary, Var, run_ga


# ============================================================================
# Equations and input ranges
# ============================================================================

SEED, SAMPLE_SIZE, SEEDS = 1, 512, list(range(30))

UNARY_REGISTRY = {
    "sin": (np.sin, sp.sin), "cos": (np.cos, sp.cos), "exp": (np.exp, sp.exp),
    "log": (np.log, sp.log), "sqrt": (np.sqrt, sp.sqrt), "abs": (np.abs, sp.Abs),
    "tanh": (np.tanh, sp.tanh),
}

# Sympy symbols for ground truth
_x, _y, _z, _r, _t, _w, _v, _m, _s, _i = sp.symbols('x y z r t w v m s i')

NGUYEN_SPECS = [
    {"id": 1, "func": lambda x: x**3 + x**2 + x, "expr": "x³+x²+x",
     "sympy": _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 2, "func": lambda x: x**4 + x**3 + x**2 + x, "expr": "x⁴+x³+x²+x",
     "sympy": _x**4 + _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 3, "func": lambda x: x**5 + x**4 + x**3 + x**2 + x, "expr": "x⁵+x⁴+x³+x²+x",
     "sympy": _x**5 + _x**4 + _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 4, "func": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x, "expr": "x⁶+x⁵+x⁴+x³+x²+x",
     "sympy": _x**6 + _x**5 + _x**4 + _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 5, "func": lambda x: np.sin(x**2) * np.cos(x) - 1, "expr": "sin(x²)cos(x)-1",
     "sympy": sp.sin(_x**2) * sp.cos(_x) - 1,
     "variables": [{"name": "x", "sampler": (-2.0, 2.0)}], "allowed_unary": ["sin", "cos"]},
    {"id": 6, "func": lambda x: np.sin(x) + np.sin(x + x**2), "expr": "sin(x)+sin(x+x²)",
     "sympy": sp.sin(_x) + sp.sin(_x + _x**2),
     "variables": [{"name": "x", "sampler": (-2.0, 2.0)}], "allowed_unary": ["sin"]},
    {"id": 7, "func": lambda x: np.log(x + 1) + np.log(x**2 + 1), "expr": "log(x+1)+log(x²+1)",
     "sympy": sp.log(_x + 1) + sp.log(_x**2 + 1),
     "variables": [{"name": "x", "sampler": (0.0, 50.0)}], "allowed_unary": ["log"]},
    {"id": 8, "func": lambda x: np.sqrt(np.abs(x)), "expr": "sqrt(|x|)",
     "sympy": sp.sqrt(sp.Abs(_x)),
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": ["abs", "sqrt"]},
    {"id": 9, "func": lambda x, y: np.sin(x) + np.sin(y**2), "expr": "sin(x)+sin(y²)",
     "sympy": sp.sin(_x) + sp.sin(_y**2),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}, {"name": "y", "sampler": (-1.0, 1.0)}],
     "allowed_unary": ["sin"]},
    {"id": 10, "func": lambda x, y: 2 * np.sin(x) * np.cos(y), "expr": "2sin(x)cos(y)",
     "sympy": 2 * sp.sin(_x) * sp.cos(_y),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}, {"name": "y", "sampler": (-1.0, 1.0)}],
     "allowed_unary": ["sin", "cos"]},
    {"id": 11, "func": lambda x, y: x**y, "expr": "x^y",
     "sympy": _x**_y,
     "variables": [{"name": "x", "sampler": (0.1, 3.0)}, {"name": "y", "sampler": (0.1, 3.0)}],
     "allowed_unary": [], "allow_pow": True},
    {"id": 12, "func": lambda x, y: x**4 - x**3 + (y**2) / 2 - y, "expr": "x⁴-x³+y²/2-y",
     "sympy": _x**4 - _x**3 + _y**2/2 - _y,
     "variables": [{"name": "x", "sampler": (-2.5, 2.5)}, {"name": "y", "sampler": (-2.5, 2.5)}],
     "allowed_unary": []},
]

# Keijzer benchmark: standard symbolic regression problems
KEIJZER_SPECS = [
    # K1: Oscillating polynomial (narrow range)
    {"id": 1, "func": lambda x: 0.3 * x * np.sin(2 * np.pi * x), "expr": "0.3x·sin(2πx)",
     "sympy": sp.Rational(3,10) * _x * sp.sin(2 * sp.pi * _x),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}], "allowed_unary": ["sin"]},
    # K2: Oscillating polynomial (medium range)
    {"id": 2, "func": lambda x: 0.3 * x * np.sin(2 * np.pi * x), "expr": "0.3x·sin(2πx)",
     "sympy": sp.Rational(3,10) * _x * sp.sin(2 * sp.pi * _x),
     "variables": [{"name": "x", "sampler": (-2.0, 2.0)}], "allowed_unary": ["sin"]},
    # K3: Oscillating polynomial (wide range)
    {"id": 3, "func": lambda x: 0.3 * x * np.sin(2 * np.pi * x), "expr": "0.3x·sin(2πx)",
     "sympy": sp.Rational(3,10) * _x * sp.sin(2 * sp.pi * _x),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}], "allowed_unary": ["sin"]},
    # K4: Complex trigonometric with exponential decay
    {"id": 4, "func": lambda x: x**3 * np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1),
     "expr": "x³e⁻ˣcos(x)sin(x)(sin²(x)cos(x)-1)",
     "sympy": _x**3 * sp.exp(-_x) * sp.cos(_x) * sp.sin(_x) * (sp.sin(_x)**2 * sp.cos(_x) - 1),
     "variables": [{"name": "x", "sampler": (0.0, 10.0)}], "allowed_unary": ["sin", "cos", "exp"]},
    # K5: Rational function (3 variables) - simplified version
    {"id": 5, "func": lambda x, y, z: (30 * x * z) / ((x - 10) * y**2 + 1e-6),
     "expr": "30xz/((x-10)y²)",
     "sympy": 30 * _x * _z / ((_x - 10) * _y**2),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}, {"name": "y", "sampler": (1.0, 2.0)},
                   {"name": "z", "sampler": (-1.0, 1.0)}], "allowed_unary": []},
    # K6: Harmonic series approximation (use continuous approx: ln(x) + gamma)
    {"id": 6, "func": lambda x: np.log(x) + 0.5772156649, "expr": "ln(x)+γ",
     "sympy": sp.log(_x) + sp.Float(0.5772156649),
     "variables": [{"name": "x", "sampler": (1.0, 50.0)}], "allowed_unary": ["log"]},
    # K7: Natural logarithm
    {"id": 7, "func": lambda x: np.log(x), "expr": "ln(x)",
     "sympy": sp.log(_x),
     "variables": [{"name": "x", "sampler": (1.0, 100.0)}], "allowed_unary": ["log"]},
    # K8: Square root
    {"id": 8, "func": lambda x: np.sqrt(x), "expr": "√x",
     "sympy": sp.sqrt(_x),
     "variables": [{"name": "x", "sampler": (0.0, 100.0)}], "allowed_unary": ["sqrt"]},
    # K9: Inverse hyperbolic sine (arcsinh)
    {"id": 9, "func": lambda x: np.arcsinh(x), "expr": "arcsinh(x)",
     "sympy": sp.asinh(_x),
     "variables": [{"name": "x", "sampler": (0.0, 100.0)}], "allowed_unary": ["log", "sqrt"]},
    # K10: Power function
    {"id": 10, "func": lambda x, y: x**y, "expr": "x^y",
     "sympy": _x**_y,
     "variables": [{"name": "x", "sampler": (0.1, 1.0)}, {"name": "y", "sampler": (0.1, 1.0)}],
     "allowed_unary": [], "allow_pow": True},
    # K11: Product with trigonometric
    {"id": 11, "func": lambda x, y: x * y + np.sin((x - 1) * (y - 1)), "expr": "xy+sin((x-1)(y-1))",
     "sympy": _x * _y + sp.sin((_x - 1) * (_y - 1)),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": ["sin"]},
    # K12: Polynomial (same as Nguyen-12)
    {"id": 12, "func": lambda x, y: x**4 - x**3 + y**2/2 - y, "expr": "x⁴-x³+y²/2-y",
     "sympy": _x**4 - _x**3 + _y**2/2 - _y,
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": []},
    # K13: Scaled trigonometric product
    {"id": 13, "func": lambda x, y: 6 * np.sin(x) * np.cos(y), "expr": "6sin(x)cos(y)",
     "sympy": 6 * sp.sin(_x) * sp.cos(_y),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": ["sin", "cos"]},
    # K14: Rational function (2D)
    {"id": 14, "func": lambda x, y: 8 / (2 + x**2 + y**2), "expr": "8/(2+x²+y²)",
     "sympy": 8 / (2 + _x**2 + _y**2),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": []},
    # K15: Mixed polynomial
    {"id": 15, "func": lambda x, y: x**3/5 + y**3/2 - y - x, "expr": "x³/5+y³/2-y-x",
     "sympy": _x**3/5 + _y**3/2 - _y - _x,
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": []},
]


def select_specs(benchmark: str):
    """The specs and display name of a benchmark: "nguyen", "keijzer" or "all".
    In "all", Keijzer ids are offset by 100 to avoid colliding with Nguyen's."""
    if benchmark == "nguyen":
        return NGUYEN_SPECS, "NGUYEN"
    if benchmark == "keijzer":
        return KEIJZER_SPECS, "KEIJZER"
    specs = NGUYEN_SPECS + [
        {**s, "id": s["id"] + 100, "expr": f"K{s['id']}:{s['expr']}"} for s in KEIJZER_SPECS
    ]
    return specs, "NGUYEN+KEIJZER"


# ============================================================================
# Edit distance to the target equation
# ============================================================================

# Note: this is the paper's simplified measure, an edit distance between the two trees'
# post-order label sequences, not a full Zhang-Shasha tree edit distance. All constants
# count as the same label.

def _get_label(node: Node) -> str:
    """Get canonical label for a node."""
    if isinstance(node, Const):
        return "CONST"  # Treat all constants as equivalent
    elif isinstance(node, Var):
        return f"VAR:{node.name}"
    elif isinstance(node, Unary):
        return f"UNARY:{node.func_name}"
    elif isinstance(node, Binary):
        return f"BINARY:{node.func_name}"
    return "UNKNOWN"


def _get_children(node: Node) -> List[Node]:
    """Get children of a node."""
    if isinstance(node, Const) or isinstance(node, Var):
        return []
    elif isinstance(node, Unary):
        return [node.child]
    elif isinstance(node, Binary):
        return [node.left, node.right]
    return []


def _simple_ted(node1: Node, node2: Node) -> int:
    """Simplified Tree Edit Distance using dynamic programming."""
    def get_nodes_postorder(node: Node) -> List[Node]:
        result = []
        stack = [(node, False)]
        while stack:
            n, visited = stack.pop()
            if visited:
                result.append(n)
            else:
                stack.append((n, True))
                for child in reversed(_get_children(n)):
                    stack.append((child, False))
        return result

    nodes1 = get_nodes_postorder(node1)
    nodes2 = get_nodes_postorder(node2)
    n1, n2 = len(nodes1), len(nodes2)

    # Simple DP: dp[i][j] = TED between first i nodes of tree1 and first j nodes of tree2
    # This is a simplification - full Zhang-Shasha is more complex
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(n1 + 1):
        dp[i][0] = i  # Delete all nodes
    for j in range(n2 + 1):
        dp[0][j] = j  # Insert all nodes

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            label1, label2 = _get_label(nodes1[i-1]), _get_label(nodes2[j-1])
            cost = 0 if label1 == label2 else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Delete from tree1
                dp[i][j-1] + 1,      # Insert into tree1
                dp[i-1][j-1] + cost  # Match/replace
            )

    return dp[n1][n2]


def compute_ted(evolved: Node, target: Node) -> int:
    """Compute Tree Edit Distance between evolved expression and target."""
    if evolved is None or target is None:
        return -1
    try:
        return _simple_ted(evolved, target)
    except Exception:
        return -1


def target_to_node(expr: sp.Expr, var_names: List[str]) -> Node:
    """Convert a target equation's sympy expression to a Node tree."""
    if isinstance(expr, sp.Number):
        return Const(float(expr))
    elif isinstance(expr, sp.Symbol):
        return Var(str(expr))
    elif isinstance(expr, sp.Add):
        args = list(expr.args)
        result = target_to_node(args[0], var_names)
        for a in args[1:]:
            result = Binary("add", result, target_to_node(a, var_names))
        return result
    elif isinstance(expr, sp.Mul):
        args = list(expr.args)
        result = target_to_node(args[0], var_names)
        for a in args[1:]:
            result = Binary("mul", result, target_to_node(a, var_names))
        return result
    elif isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == sp.Rational(1, 2):
            return Unary("sqrt", target_to_node(base, var_names))
        elif exp == -1:
            return Binary("div", Const(1.0), target_to_node(base, var_names))
        return Binary("pow", target_to_node(base, var_names), target_to_node(exp, var_names))
    elif isinstance(expr, sp.sin):
        return Unary("sin", target_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.cos):
        return Unary("cos", target_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.exp):
        return Unary("exp", target_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.log):
        return Unary("log", target_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.sqrt):
        return Unary("sqrt", target_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.Abs):
        return Unary("abs", target_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.tanh):
        return Unary("tanh", target_to_node(expr.args[0], var_names))
    else:
        # Fallback: try to expand and retry
        expanded = sp.expand(expr)
        if expanded != expr:
            return target_to_node(expanded, var_names)
        return Const(0.0)  # Fallback


# ============================================================================
# One run
# ============================================================================

# GA settings shared by both modes
ga_config = {
    "pop_size": 300, "gens": 200, "max_init_depth": 6, "max_depth": 30,
    "crossover_rate": 0.85, "mutation_rate": 0.5, "tournament_k": 9, "elitism": 24,
    "use_learned_crossover": False, "selector_lr": 0.001,
    "mlp_layers": 16, "mlp_hidden_dim": 64, "mlp_activation": "leaky_relu", "mlp_dropout": 0.1,
    "use_critic": True, "stopping_metric": "r2", "stopping_threshold": 1.0,
    "complexity_coef": 1e-4, "inclusion_coef": 1.0, "prob_const": 0.3, "const_range": (-5, 5),
    "point_mutation_prob": 0.1, "simplify_new_genomes": False, "simplify_timeout": 0.2,
    "selector_update_freq": 1, "temperature": 0.7, "entropy_weight": 0.1,
    "opt_constants": True, "const_opt_freq": 10, "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 100, "verbose": True, "use_pareto": True, "print_freq": 10, "seed": SEED,
}
# GP-SR-RLC: the learned crossover and its selector settings
learned_overrides = {
    "use_learned_crossover": True, "mlp_layers": 4, "mlp_hidden_dim": 64, "mlp_dropout": 0.0,
    "temperature": 1.0, "entropy_weight": 0.08, "heuristic_bias_scale": 0.08,
    "selector_update_freq": 1, "selector_warmup_gens": 30, "selector_recent_gen_window": 50,
    "selector_min_history": 64, "selector_reward_clip": 3.0, "selector_lr": 3e-4,
    "use_critic": True, "selector_device": "cpu",
}
# Baseline: random subtree crossover
random_overrides = {"use_learned_crossover": False, "selector_device": "cpu"}


@dataclass(frozen=True)
class Task:
    mode: str
    spec_id: int
    seed: int
    benchmark: str = "nguyen"


def run_single_task(t: Task) -> Dict:
    # Select correct specs based on benchmark
    if t.benchmark == "nguyen":
        specs_list = NGUYEN_SPECS
    elif t.benchmark == "keijzer":
        specs_list = KEIJZER_SPECS
    else:  # all
        specs_list = NGUYEN_SPECS + [{**s, "id": s["id"] + 100} for s in KEIJZER_SPECS]
    spec = next((s for s in specs_list if s["id"] == t.spec_id), None)
    if spec is None:
        return {"mode": t.mode, "spec_id": t.spec_id, "seed": t.seed, "expr_target": "?",
                "n_vars": 0, "status": "error", "best_r2": float("nan"),
                "best_expr": "", "elapsed_sec": 0.0, "error": f"spec_id {t.spec_id} not found"}
    started = time.time()
    status, error, best_expr, best_r2 = "ok", "", "", float("nan")
    try:
        rng = np.random.default_rng(t.seed + t.spec_id * 1000)
        vnames = [v["name"] for v in spec["variables"]]
        cols = [rng.uniform(v["sampler"][0], v["sampler"][1], size=SAMPLE_SIZE) for v in spec["variables"]]
        try:
            y = spec["func"](*cols)
        except Exception:
            y = np.array([spec["func"](*[c[i] for c in cols]) for i in range(SAMPLE_SIZE)])
        X = np.column_stack(cols)
        y = np.asarray(y, dtype=float).ravel()
        ok = np.isfinite(y)
        if not ok.all():
            X, y = X[ok], y[ok]
        if len(y) < 10:
            raise ValueError(f"Too few valid samples ({len(y)}) for Nguyen-{spec['id']}")
        allowed = spec.get("allowed_unary") or []
        unary = {n: UNARY_REGISTRY[n] for n in allowed if n in UNARY_REGISTRY}
        kw = {**ga_config, **(learned_overrides if t.mode == "learned" else random_overrides),
              "seed": t.seed, "allow_pow": bool(spec.get("allow_pow", False))}
        best, r2v = run_ga(X, y, var_names=vnames, unary_funcs=unary, **kw)
        best_r2 = float(r2v) if r2v is not None else float("nan")
        best_expr = best.to_string() if best is not None else ""
        # Compute TED if ground truth sympy is available
        if best is not None and "sympy" in spec:
            try:
                target_node = target_to_node(spec["sympy"], vnames)
                ted = compute_ted(best, target_node)
            except Exception:
                ted = -1
        else:
            ted = -1
    except Exception as exc:
        status, error, ted = "error", str(exc), -1
    return {"mode": t.mode, "spec_id": t.spec_id, "seed": t.seed, "expr_target": spec["expr"],
            "n_vars": len(spec["variables"]), "status": status, "best_r2": best_r2,
            "best_expr": best_expr, "ted": ted, "elapsed_sec": time.time() - started, "error": error}
