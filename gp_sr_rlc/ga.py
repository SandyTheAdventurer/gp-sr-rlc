"""The genetic programming loop for symbolic regression."""
import random

import numpy as np
from tqdm.auto import tqdm

from . import crossover as lx
from . import gp


def run_ga(
    x,
    y: np.ndarray,
    pop_size=300,
    gens=50,
    max_init_depth=4,
    max_depth=7,
    crossover_rate=0.7,
    mutation_rate=0.3,
    tournament_k=3,
    elitism=1,
    var_names=None,
    unary_funcs=None,
    use_learned_crossover=True,
    selector_lr=0.001,
    use_transformer=False,
    transformer_heads=4,
    transformer_layers=2,
    mlp_layers=3,
    mlp_hidden_dim=64,
    mlp_activation='relu',
    mlp_dropout=0.0,
    use_critic=True,
    stopping_metric='mse',
    stopping_threshold=None,
    complexity_coef=1e-3,
    inclusion_coef=1.0,
    prob_const=0.3,
    const_range=(-5, 5),
    point_mutation_prob=0.05,
    simplify_new_genomes=False,
    simplify_timeout=0.2,
    selector_update_freq=5,
    selector_warmup_gens=8,
    selector_recent_gen_window=2,
    selector_min_history=64,
    selector_reward_clip=5.0,
    temperature=0.5,
    critic_weight=0.5,
    entropy_weight=0.01,
    use_pareto=False,
    opt_constants=False,
    const_opt_freq=5,
    const_opt_method='L-BFGS-B',
    const_opt_maxiter=50,
    verbose=False,
    print_freq=1,
    seed=gp.DEFAULT_SEED,
    metrics_callback=None,
    allow_pow=False,
    selector_device="auto",
    heuristic_bias_scale=1.0,
):
    """Evolve an expression for y from x. Returns (best_tree, best_r2)."""
    if seed is None:
        seed = gp.DEFAULT_SEED
    gp.set_global_seed(seed)

    inputs, resolved_names = gp.prepare_inputs(x, var_names)
    gp.set_variables(resolved_names)

    target_device = gp.resolve_selector_device(use_learned_crossover, selector_device)
    lx.set_selector_device(target_device)
    if verbose and use_learned_crossover:
        print(f"Selector device: {gp.DEVICE}")

    gp.configure_operators(unary_funcs, allow_pow)
    lx.clear_history()

    init_max_depth = max(1, max_init_depth)
    pop = [
        gp.random_tree(
            max_depth=random.randint(1, init_max_depth),
            prob_const=prob_const,
            const_range=const_range,
        )
        for _ in range(pop_size)
    ]
    if simplify_new_genomes:
        pop = [gp.simplify_node_with_timeout(ind, simplify_timeout) for ind in pop]

    lx.reset_selector(
        use_learned_crossover,
        selector_lr=selector_lr,
        mlp_hidden_dim=mlp_hidden_dim,
        use_transformer=use_transformer,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        mlp_layers=mlp_layers,
        mlp_activation=mlp_activation,
        mlp_dropout=mlp_dropout,
        use_critic=use_critic,
        verbose=verbose,
    )

    # Validate stopping criterion
    if stopping_threshold is not None:
        if stopping_metric not in ['mse', 'r2', None]:
            raise ValueError("stopping_metric must be 'mse', 'r2', or None")
        if stopping_metric == 'r2' and (stopping_threshold < 0 or stopping_threshold > 1):
            print(f"Warning: R² stopping threshold {stopping_threshold} is outside typical range [0, 1]")
        if verbose:
            print(f"Stopping criterion: {stopping_metric} <= {stopping_threshold}" if stopping_metric == 'mse' else f"Stopping criterion: {stopping_metric} >= {stopping_threshold}")

    y = np.asarray(y, dtype=float).reshape(-1)
    n_samples = len(next(iter(inputs.values())))
    if y.shape[0] != n_samples:
        raise ValueError("y must have the same length as the input samples.")

    def evaluate_r2(individual):
        try:
            preds = individual.evaluate(inputs)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return None
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else None
        except Exception:
            return None

    best = None
    best_score = float('inf')
    best_r2 = None

    gen_iterator = tqdm(range(gens), desc="GA generations", unit="gen")
    for gen in gen_iterator:
        scores = [gp.fitness(ind, inputs, y, complexity_coef, inclusion_coef) for ind in pop]
        if use_pareto:
            pareto_objectives_list = [
                gp.pareto_objectives(ind, inputs, y, inclusion_coef)
                for ind in pop
            ]
            pareto_ranks, pareto_crowding = gp.pareto_rank_and_crowding(pareto_objectives_list)
        else:
            pareto_ranks, pareto_crowding = None, None

        # Update selector every few generations
        if (
            use_learned_crossover
            and gen >= max(1, int(selector_warmup_gens))
            and gen % selector_update_freq == 0
        ):
            lx.update_selector(
                pop,
                scores,
                inputs,
                y,
                complexity_coef,
                inclusion_coef,
                critic_weight=critic_weight,
                entropy_weight=entropy_weight,
                recent_gen_window=selector_recent_gen_window,
                min_history=selector_min_history,
                reward_clip=selector_reward_clip,
                current_gen=gen,
                temperature=temperature,
                heuristic_bias_scale=heuristic_bias_scale,
            )

        # track best
        for ind, s in zip(pop, scores):
            if s < best_score:
                best_score = s
                best = ind.clone()
                r2_val = evaluate_r2(best)
                if r2_val is not None:
                    best_r2 = r2_val

        # Check stopping criterion
        if stopping_threshold is not None and best is not None:
            try:
                y_pred = best.evaluate(inputs)
                if not (np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred))):
                    if stopping_metric == 'mse':
                        current_mse = float(np.mean((y - y_pred)**2))
                        if current_mse <= stopping_threshold:
                            if verbose:
                                print(f"\nStopping criterion reached at generation {gen+1}")
                                print(f"MSE: {current_mse:.6g} <= {stopping_threshold}")
                            break
                    elif stopping_metric == 'r2':
                        ss_res = np.sum((y - y_pred)**2)
                        ss_tot = np.sum((y - np.mean(y))**2)
                        current_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                        if current_r2 >= stopping_threshold:
                            if verbose:
                                print(f"\nStopping criterion reached at generation {gen+1}")
                                print(f"R²: {current_r2:.6g} >= {stopping_threshold}")
                            break
            except Exception:
                pass  # Continue if evaluation fails

        next_pop = []
        # elitism
        if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
            sorted_idx = sorted(
                range(len(pop)),
                key=lambda i: (pareto_ranks[i], -pareto_crowding[i], scores[i])
            )
        else:
            sorted_idx = sorted(range(len(pop)), key=lambda i: scores[i])
        # Optimize constants only for elites (optional, periodic)
        optimize_elites = opt_constants and gen > 0 and gen % const_opt_freq == 0
        if optimize_elites and verbose:
            print(f"\nOptimizing constants for elites at generation {gen+1}...")
        for i in range(elitism):
            elite = pop[sorted_idx[i]].clone()
            if optimize_elites:
                elite = gp.optimize_constants(
                    elite,
                    inputs,
                    y,
                    method=const_opt_method,
                    maxiter=const_opt_maxiter,
                )
            if simplify_new_genomes:
                elite = gp.simplify_node_with_timeout(elite, simplify_timeout)
            next_pop.append(elite)
        # produce rest
        while len(next_pop) < pop_size:
            use_learned_now = use_learned_crossover and gen >= max(1, int(selector_warmup_gens))
            if random.random() < crossover_rate:
                try:
                    if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
                        parent1 = gp.tournament_select_pareto(pop, pareto_ranks, pareto_crowding, scores, tournament_k)
                    else:
                        parent1 = gp.tournament_select(pop, scores, tournament_k)
                except RecursionError:
                    parent1 = random.choice(pop).clone()
                try:
                    if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
                        parent2 = gp.tournament_select_pareto(pop, pareto_ranks, pareto_crowding, scores, tournament_k)
                    else:
                        parent2 = gp.tournament_select(pop, scores, tournament_k)
                except RecursionError:
                    parent2 = random.choice(pop).clone()
                child1, child2, _ = lx.learned_subtree_crossover(
                    parent1,
                    parent2,
                    use_learned=use_learned_now,
                    temperature=temperature,
                    record_gen=gen,
                    heuristic_bias_scale=heuristic_bias_scale,
                )
                # optionally mutate children
                if random.random() < mutation_rate:
                    child1 = gp.subtree_mutation(child1, max_depth, prob_const, const_range)
                if random.random() < mutation_rate:
                    child2 = gp.subtree_mutation(child2, max_depth, prob_const, const_range)
                if simplify_new_genomes:
                    child1 = gp.simplify_node_with_timeout(child1, simplify_timeout)
                    child2 = gp.simplify_node_with_timeout(child2, simplify_timeout)
                # Reject bloated offspring: fall back to the parent when depth exceeds max_depth
                if gp.too_big(child1, max_depth):
                    child1 = parent1.clone()
                if gp.too_big(child2, max_depth):
                    child2 = parent2.clone()
                next_pop.append(child1)
                if len(next_pop) < pop_size:
                    next_pop.append(child2)
            else:
                try:
                    if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
                        parent = gp.tournament_select_pareto(pop, pareto_ranks, pareto_crowding, scores, tournament_k)
                    else:
                        parent = gp.tournament_select(pop, scores, tournament_k)
                except RecursionError:
                    parent = random.choice(pop).clone()
                child = parent
                if random.random() < mutation_rate:
                    child = gp.subtree_mutation(child, max_depth, prob_const, const_range)
                # small point mutations
                child = gp.point_mutation(child, prob_change=point_mutation_prob)
                if simplify_new_genomes:
                    child = gp.simplify_node_with_timeout(child, simplify_timeout)
                if gp.too_big(child, max_depth):
                    child = parent.clone()
                next_pop.append(child)
        pop = next_pop

        if verbose and ((gen + 1) % print_freq == 0 or gen == 0):
            # Calculate R² for the best individual
            try:
                y_pred_best = best.evaluate(inputs)
                if not (np.any(np.isinf(y_pred_best)) or np.any(np.isnan(y_pred_best))):
                    ss_res = np.sum((y - y_pred_best)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2_best = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g} R²: {r2_best:.6f} expr: {best.to_string()}")
                else:
                    print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g} expr: {best.to_string()}")
            except Exception:
                print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g} expr: {best.to_string()}")

        if metrics_callback is not None and best is not None:
            try:
                y_pred = np.asarray(best.evaluate(inputs), dtype=float)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    raise ValueError("Invalid predictions for metrics callback.")
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
            except Exception:
                r2 = float("nan")
            metrics_callback(gen, float(r2))

    # best_r2 is only refreshed when the new best evaluates cleanly; report the final best's own
    if best is not None:
        best_r2 = evaluate_r2(best)

    return best, best_r2
