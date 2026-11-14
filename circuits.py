"""Resistor-network application: evolve a network that meets a target resistance, with random
or RL-guided crossover. Node 0 is V+ and node 1 is GND; the other nodes are internal.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from gp_sr_rlc import DEVICE, SubtreeSelector, set_global_seed


# ============================================================================
# Networks and DC analysis
# ============================================================================

@dataclass
class Resistor:
    n1: int
    n2: int
    value: float  # Ohms


@dataclass
class Topology:
    resistors: list[Resistor]
    num_nodes: int

    def clone(self) -> Topology:
        return copy.deepcopy(self)

    def __str__(self) -> str:
        r = ", ".join(f"R({b.n1}-{b.n2}:{b.value:.1f}Ω)" for b in self.resistors)
        return f"Topology(n={self.num_nodes}; {r})"


def signature(top: Topology) -> tuple:
    """A hashable key for a topology (used to cache evaluations)."""
    return (top.num_nodes,
            tuple(sorted((r.n1, r.n2, round(r.value, 1)) for r in top.resistors)))


def solve_dc(top: Topology, v_in: float = 1.0) -> float | None:
    """
    Apply v_in between node 0 (V+) and node 1 (GND).
    Return equivalent resistance seen from input, or None if failed.

    Node 0 = V+ (voltage source)
    Node 1 = GND (reference)
    """
    if top.num_nodes < 2:
        return None

    # Build conductance matrix for nodes 2..N-1 (node 0 is V+, node 1 is GND)
    unknowns = list(range(2, top.num_nodes))
    if not unknowns:
        # Only 2 nodes: direct connection
        g_total = sum(1.0 / max(r.value, 1e-9) for r in top.resistors
                      if {r.n1, r.n2} == {0, 1})
        if g_total < 1e-12:
            return None
        return 1.0 / g_total

    nmap = {n: i for i, n in enumerate(unknowns)}
    nvar = len(unknowns)

    G = np.zeros((nvar, nvar))
    I = np.zeros(nvar)

    # Fixed voltages
    v_fixed = {0: v_in, 1: 0.0}

    for r in top.resistors:
        g = 1.0 / max(r.value, 1e-9)
        n1, n2 = r.n1, r.n2

        i1 = nmap.get(n1)
        i2 = nmap.get(n2)

        if i1 is not None:
            G[i1, i1] += g
            if i2 is not None:
                G[i1, i2] -= g
            elif n2 in v_fixed:
                I[i1] += g * v_fixed[n2]

        if i2 is not None:
            G[i2, i2] += g
            if i1 is not None:
                G[i2, i1] -= g
            elif n1 in v_fixed:
                I[i2] += g * v_fixed[n1]

    # Solve for node voltages
    try:
        v_nodes = np.linalg.solve(G, I)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(v_nodes)):
        return None

    # Calculate current from V+ node
    v_all = np.zeros(top.num_nodes)
    v_all[0] = v_in
    v_all[1] = 0.0
    for n, i in nmap.items():
        v_all[n] = v_nodes[i]

    i_total = 0.0
    for r in top.resistors:
        if r.n1 == 0:
            i_total += (v_all[0] - v_all[r.n2]) / max(r.value, 1e-9)
        elif r.n2 == 0:
            i_total += (v_all[0] - v_all[r.n1]) / max(r.value, 1e-9)

    if abs(i_total) < 1e-12:
        return None

    return v_in / i_total


def prune_unused(top: Topology) -> Topology:
    """Remove resistors not on any V+ to GND path."""
    # BFS to find nodes reachable from V+ (node 0)
    adj = {i: set() for i in range(top.num_nodes)}
    for r in top.resistors:
        adj[r.n1].add(r.n2)
        adj[r.n2].add(r.n1)

    def reachable(start: int) -> set:
        visited = {start}
        queue = [start]
        while queue:
            n = queue.pop(0)
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited

    from_vplus = reachable(0)
    from_gnd = reachable(1)
    useful_nodes = from_vplus & from_gnd

    # Keep only resistors where both nodes are useful
    kept = [r for r in top.resistors if r.n1 in useful_nodes and r.n2 in useful_nodes]
    if not kept:
        return top  # Can't prune everything

    # Renumber nodes
    old_to_new = {0: 0, 1: 1}
    next_id = 2
    for r in kept:
        for n in (r.n1, r.n2):
            if n not in old_to_new:
                old_to_new[n] = next_id
                next_id += 1

    pruned = Topology(
        [Resistor(old_to_new[r.n1], old_to_new[r.n2], r.value) for r in kept],
        next_id
    )
    return pruned


# ============================================================================
# Objectives and Pareto ranking
# ============================================================================

TARGET_RESISTANCE = 100.0  # Target: 100 ohms


def evaluate_pareto(top: Topology, target: float = TARGET_RESISTANCE) -> tuple[float, int]:
    """Return (error, num_resistors) for Pareto ranking."""
    pruned = prune_unused(top)
    r_eq = solve_dc(pruned)
    if r_eq is None or r_eq <= 0:
        return (1e6, len(top.resistors))
    error = abs(r_eq - target) / target
    return (error, len(pruned.resistors))


def dominates(a: tuple[float, int], b: tuple[float, int]) -> bool:
    """True if a dominates b (better or equal in all, strictly better in one)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def pareto_rank(objs: list[tuple[float, int]]) -> list[int]:
    """Simple non-dominated sorting. Returns rank for each individual (0 = best)."""
    n = len(objs)
    dom_count = [sum(1 for j in range(n) if i != j and dominates(objs[j], objs[i])) for i in range(n)]

    ranks = [0] * n
    assigned = [False] * n
    rank = 0
    remaining = n

    while remaining > 0:
        for i in range(n):
            if not assigned[i] and dom_count[i] == 0:
                ranks[i] = rank
                assigned[i] = True
                remaining -= 1
        # Decrease counts for next round
        for i in range(n):
            if assigned[i] and ranks[i] == rank:
                for j in range(n):
                    if not assigned[j] and dominates(objs[i], objs[j]):
                        dom_count[j] -= 1
        rank += 1

    return ranks


def crowding_distance(objs: list[tuple[float, int]], indices: list[int]) -> dict[int, float]:
    """Crowding distance for diversity preservation."""
    if len(indices) <= 2:
        return {i: float('inf') for i in indices}

    dist = {i: 0.0 for i in indices}
    for dim in range(2):
        s = sorted(indices, key=lambda i: objs[i][dim])
        dist[s[0]] = dist[s[-1]] = float('inf')
        rng = objs[s[-1]][dim] - objs[s[0]][dim]
        if rng > 1e-12:
            for k in range(1, len(s) - 1):
                dist[s[k]] += (objs[s[k+1]][dim] - objs[s[k-1]][dim]) / rng
    return dist


def evaluate(top: Topology, target: float = TARGET_RESISTANCE) -> float:
    """Single objective for backward compat (weighted sum)."""
    err, comp = evaluate_pareto(top, target)
    return err + 0.01 * comp


# ============================================================================
# Random networks, mutation and random crossover
# ============================================================================

R_VALUES = [10, 22, 47, 100]  # Standard resistor values


def _rand_pair(n: int) -> tuple[int, int]:
    a = random.randrange(n)
    b = random.randrange(n - 1)
    return (a, b + 1) if b >= a else (a, b)


def repair(top: Topology):
    """Reconnect self-loops and order each resistor's nodes (n1 < n2)."""
    for r in top.resistors:
        if r.n1 == r.n2:
            r.n1, r.n2 = _rand_pair(top.num_nodes)
        if r.n1 > r.n2:
            r.n1, r.n2 = r.n2, r.n1


def random_topology(nodes: int = 4, nr: int = 4) -> Topology:
    top = Topology(
        [Resistor(*_rand_pair(nodes), random.choice(R_VALUES)) for _ in range(nr)],
        nodes,
    )
    repair(top)
    return top


def mutate(top: Topology, rate: float = 0.3) -> Topology:
    out = top.clone()
    for r in out.resistors:
        if random.random() < rate:
            # Mutate connection
            r.n1, r.n2 = _rand_pair(out.num_nodes)
        if random.random() < rate:
            # Mutate value
            r.value = random.choice(R_VALUES)
    repair(out)
    return out


def crossover(a: Topology, b: Topology) -> tuple[Topology, Topology]:
    c1, c2 = a.clone(), b.clone()
    for i in range(min(len(c1.resistors), len(c2.resistors))):
        if random.random() < 0.5:
            c1.resistors[i], c2.resistors[i] = c2.resistors[i], c1.resistors[i]
    repair(c1)
    repair(c2)
    return c1, c2


# ============================================================================
# RL-guided crossover
# ============================================================================

FEATURE_DIM = 8


def _resistor_features(top: Topology, r: Resistor, idx: int) -> torch.Tensor:
    deg = np.bincount([res.n1 for res in top.resistors] + [res.n2 for res in top.resistors],
                      minlength=top.num_nodes).astype(float)
    maxd = max(deg.max(), 1)
    return torch.tensor([
        r.n1 / max(top.num_nodes - 1, 1), r.n2 / max(top.num_nodes - 1, 1),
        deg[r.n1] / maxd, deg[r.n2] / maxd,
        r.n1 in {0, 1} or r.n2 in {0, 1},
        np.log10(r.value) / 3, idx / max(len(top.resistors) - 1, 1), len(top.resistors) / 10,
    ], dtype=torch.float32, device=DEVICE)


def learned_crossover(a: Topology, b: Topology, selector: SubtreeSelector, temp: float,
                      history: list | None = None) -> tuple[Topology, Topology]:
    c1, c2 = a.clone(), b.clone()
    if not c1.resistors or not c2.resistors:
        return c1, c2

    feat_a = torch.stack([_resistor_features(c1, r, i) for i, r in enumerate(c1.resistors)])
    feat_b = torch.stack([_resistor_features(c2, r, i) for i, r in enumerate(c2.resistors)])

    with torch.no_grad():
        logits_a, _ = selector(feat_a, return_value=True)
        logits_b, _ = selector(feat_b, return_value=True)
        pair = torch.clamp((logits_a.unsqueeze(1) + logits_b.unsqueeze(0)) / max(temp, 1e-6), -10, 10)
        idx = int(torch.multinomial(torch.softmax(pair.flatten(), 0), 1).item())
        ai, bi = idx // len(c2.resistors), idx % len(c2.resistors)

    c1.resistors[ai], c2.resistors[bi] = c2.resistors[bi], c1.resistors[ai]
    repair(c1); repair(c2)

    if history is not None:
        history.append({"feat_a": feat_a, "feat_b": feat_b, "ai": ai, "bi": bi,
                        "c1": c1.clone(), "c2": c2.clone()})
    return c1, c2


def update_selector(selector: SubtreeSelector, optimizer: optim.Optimizer,
                    history: list, eval_fn) -> None:
    if not history:
        return

    # Compute rewards
    base = eval_fn(history[0]["c1"])
    rewards = torch.tensor([base - min(eval_fn(r["c1"]), eval_fn(r["c2"])) for r in history],
                           dtype=torch.float32, device=DEVICE)
    if rewards.std() < 1e-12:
        history.clear()
        return
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Compute log probs
    log_probs = []
    for rec in history:
        logits_a, _ = selector(rec["feat_a"].to(DEVICE), return_value=True)
        logits_b, _ = selector(rec["feat_b"].to(DEVICE), return_value=True)
        pair = (logits_a.unsqueeze(1) + logits_b.unsqueeze(0)).flatten()
        idx = rec["ai"] * rec["feat_b"].shape[0] + rec["bi"]
        log_probs.append(torch.log_softmax(pair, dim=0)[idx])

    loss = -(torch.stack(log_probs) * rewards).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
    optimizer.step()
    history.clear()


# ============================================================================
# Evolution
# ============================================================================

def evolve(
    target: float = TARGET_RESISTANCE,
    pop_size: int = 100,
    gens: int = 30,
    nodes: int = 4,
    nr: int = 4,
    cx_rate: float = 0.85,
    mut_rate: float = 0.3,
    tour_k: int = 3,
    elite: int = 2,
    learned: bool = False,
    sel_lr: float = 3e-4,
    sel_freq: int = 3,
    temp: float = 0.6,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[Topology, float, list[Topology]]:
    """Returns (best, best_score, pareto_front)."""
    set_global_seed(seed)
    cache: dict[tuple, tuple[float, int]] = {}

    def eval_cached(top: Topology) -> tuple[float, int]:
        key = signature(top)
        if key not in cache:
            cache[key] = evaluate_pareto(top, target)
        return cache[key]

    def scalar_score(top: Topology) -> float:
        err, comp = eval_cached(top)
        return err + 0.01 * comp

    pop = [random_topology(nodes, nr) for _ in range(pop_size)]

    selector = history = optimizer = None
    if learned:
        selector = SubtreeSelector(feature_dim=FEATURE_DIM, hidden_dim=32,
                                   mlp_layers=2, use_critic=True).to(DEVICE)
        optimizer = optim.Adam(selector.parameters(), lr=sel_lr)
        history = []

    best, best_score = pop[0], float("inf")

    for gen in range(gens):
        objs = [eval_cached(t) for t in pop]
        ranks = pareto_rank(objs)

        # Track best
        for i, t in enumerate(pop):
            score = objs[i][0] + 0.01 * objs[i][1]
            if score < best_score:
                best_score, best = score, t.clone()

        if verbose and (gen == 0 or (gen + 1) % 5 == 0):
            pruned = prune_unused(best)
            print(f"Gen {gen+1:3d} | error={eval_cached(best)[0]:.4f} | R={len(pruned.resistors)} | R_eq={solve_dc(pruned):.1f}Ω | front={ranks.count(0)}")

        if learned and history and gen % sel_freq == 0:
            update_selector(selector, optimizer, history, scalar_score)

        # Crowding for diversity
        crowding = {}
        for r in range(max(ranks) + 1):
            crowding.update(crowding_distance(objs, [i for i in range(pop_size) if ranks[i] == r]))

        # Selection key: rank first, then crowding (higher = better)
        key = lambda i: (ranks[i], -crowding.get(i, 0))
        next_pop = [pop[i].clone() for i in sorted(range(pop_size), key=key)[:elite]]

        def select():
            cands = random.sample(range(pop_size), min(tour_k, pop_size))
            return pop[min(cands, key=key)].clone()

        while len(next_pop) < pop_size:
            p1, p2 = select(), select()
            if random.random() < cx_rate:
                c1, c2 = learned_crossover(p1, p2, selector, temp, history) if learned else crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            next_pop.append(mutate(c1, mut_rate))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2, mut_rate))
        pop = next_pop

    # Final Pareto front
    final_objs = [eval_cached(t) for t in pop]
    final_ranks = pareto_rank(final_objs)
    front = {signature(prune_unused(pop[i])): prune_unused(pop[i]) for i in range(pop_size) if final_ranks[i] == 0}

    return prune_unused(best), best_score, list(front.values())
