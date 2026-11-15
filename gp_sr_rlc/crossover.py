"""GP-SR-RLC: subtree crossover whose crossover points are sampled by an actor-critic selector.

The selector, its optimizer and the record of crossovers it learns from are module state,
reset by `reset_selector` at the start of each run.
"""
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import gp
from .gp import Binary, Const, Node, Unary, Var, collect_nodes_with_parents, fitness, replace_child


# ============================================================================
# Selector network and node features
# ============================================================================

# Neural network that learns to select good subtree crossover points using A2C.
class SubtreeSelector(nn.Module):
    def __init__(self, feature_dim=10, hidden_dim=64, use_transformer=False, num_heads=4, num_layers=2,
                 mlp_layers=3, mlp_activation='relu', mlp_dropout=0.0, use_critic=True):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_critic = use_critic

        if use_transformer:
            # Transformer-based feature extractor
            self.feature_projection = nn.Linear(feature_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Actor head (policy)
            self.actor = nn.Linear(hidden_dim, 1)
            # Critic head (value function) - only if enabled
            if self.use_critic:
                self.critic = nn.Linear(hidden_dim, 1)
        else:
            # Customizable MLP-based feature extractor
            activation_map = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'leaky_relu': nn.LeakyReLU,
                'elu': nn.ELU,
                'gelu': nn.GELU,
            }

            if mlp_activation not in activation_map:
                raise ValueError(f"Unsupported activation: {mlp_activation}. Choose from {list(activation_map.keys())}")

            activation_fn = activation_map[mlp_activation]

            # Build shared MLP layers
            shared_layers = []
            input_size = feature_dim

            for i in range(mlp_layers - 1):
                shared_layers.append(nn.Linear(input_size, hidden_dim))
                shared_layers.append(activation_fn())
                if mlp_dropout > 0:
                    shared_layers.append(nn.Dropout(mlp_dropout))
                input_size = hidden_dim

            self.shared_net = nn.Sequential(*shared_layers)

            # Actor head (policy) - outputs action logits
            self.actor = nn.Linear(hidden_dim, 1)

            # Critic head (value function) - only if enabled
            if self.use_critic:
                self.critic = nn.Linear(hidden_dim, 1)

        # Move model to device
        self.to(gp.DEVICE)

    def forward(self, features, return_value=False):
        """Returns action logits and optionally value estimates.

        Args:
            features: (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
            return_value: If True, also return value estimates from critic

        Returns:
            If return_value=False: action_logits
            If return_value=True: (action_logits, values) or (action_logits, None) if no critic
        """
        if self.use_transformer:
            # If 2D, add sequence dimension
            if features.dim() == 2:
                features = features.unsqueeze(0)  # (1, seq_len, feature_dim)

            # Project features to transformer dimension
            x = self.feature_projection(features)  # (batch, seq_len, hidden_dim)

            # Apply transformer encoder
            x = self.transformer(x)  # (batch, seq_len, hidden_dim)

            # Actor and critic outputs
            action_logits = self.actor(x).squeeze(-1)  # (batch, seq_len)

            # Remove batch dimension if it was added
            if action_logits.dim() == 2 and action_logits.size(0) == 1:
                action_logits = action_logits.squeeze(0)

            if return_value:
                if self.use_critic:
                    values = self.critic(x).squeeze(-1)  # (batch, seq_len)
                    if values.dim() == 2 and values.size(0) == 1:
                        values = values.squeeze(0)
                else:
                    values = None
                return action_logits, values
            else:
                return action_logits
        else:
            # MLP path
            shared_features = self.shared_net(features)
            action_logits = self.actor(shared_features).squeeze(-1)

            if return_value:
                if self.use_critic:
                    values = self.critic(shared_features).squeeze(-1)
                else:
                    values = None
                return action_logits, values
            else:
                return action_logits


def collect_nodes_with_features(node: Node, parent=None, attr_name=None, depth=0):
    """The tree's nodes in pre-order as (node, parent, attribute name), and their feature matrix
    (one row per node): node type, relative size, depth and position, parent type, operator identity."""
    nodes, info = [], []

    def recurse(n, p, attr, d):
        """Append n's subtree; return its (size, depth)."""
        idx = len(nodes)
        nodes.append((n, p, attr))
        info.append(None)
        if isinstance(n, Unary):
            size, dep = recurse(n.child, n, 'child', d + 1)
            size, dep = 1 + size, 1 + dep
        elif isinstance(n, Binary):
            left_size, left_dep = recurse(n.left, n, 'left', d + 1)
            right_size, right_dep = recurse(n.right, n, 'right', d + 1)
            size, dep = 1 + left_size + right_size, 1 + max(left_dep, right_dep)
        else:
            size, dep = 1, 0
        info[idx] = (d, size, dep)
        return size, dep

    tree_size, tree_depth = recurse(node, parent, attr_name, depth)
    rows = []
    for (n, p, _), (depth_from_root, size, dep) in zip(nodes, info):
        features = []

        # Node type features (one-hot-ish)
        features.append(1.0 if isinstance(n, Const) else 0.0)
        features.append(1.0 if isinstance(n, Var) else 0.0)
        features.append(1.0 if isinstance(n, Unary) else 0.0)
        features.append(1.0 if isinstance(n, Binary) else 0.0)

        # Structural features
        features.append(size / max(tree_size, 1))  # Relative size
        features.append(dep / max(tree_depth, 1))  # Relative depth
        features.append(depth_from_root / max(tree_depth, 1))  # Position in tree

        # Parent features
        features.append(1.0 if p is None else 0.0)  # Is root
        features.append(1.0 if isinstance(p, Unary) else 0.0)
        features.append(1.0 if isinstance(p, Binary) else 0.0)

        # Specific operator features
        unary_one_hot = [0.0] * len(gp.ALL_UNARY)
        if isinstance(n, Unary):
            try:
                unary_one_hot[gp.ALL_UNARY.index(n.func_name)] = 1.0
            except ValueError:
                pass
        features.extend(unary_one_hot)

        binary_one_hot = [0.0] * len(gp.ALL_BINARY)
        if isinstance(n, Binary):
            try:
                binary_one_hot[gp.ALL_BINARY.index(n.func_name)] = 1.0
            except ValueError:
                pass
        features.extend(binary_one_hot)
        rows.append(features)

    return nodes, torch.tensor(rows, dtype=torch.float32, device=gp.DEVICE)


# ============================================================================
# Learned crossover and its A2C update
# ============================================================================

subtree_selector = None
selector_optimizer = None
crossover_history = []
selector_feature_dim = None

# Non-learned informed operators, for comparison (run_ga's crossover_operator): None for the default
# (random, or the learned selector), "heuristic", "ssc" or "sizefair"
OPERATORS = (None, "heuristic", "ssc", "sizefair")
operator_name = None
semantic_inputs = None  # training inputs, on which SSC measures subtree semantics
SSC_BOUNDS, SSC_TRIALS = (1e-4, 0.4), 12  # Uy et al. (2011): LBSS, UBSS and Max_Trial


# Tournaments pick the same parents many times in a generation. A tree's feature matrix depends only
# on the tree, and its node logits only on the tree and the selector's weights, so both are kept per
# tree: features until the operator set changes, logits until the weights may change.
NODE_CACHE_LIMIT = 4000
_feature_cache, _logit_cache = {}, {}
_logits_repeatable = False  # False when the selector's forward pass draws random numbers (dropout)


def _clear_node_caches(features=True) -> None:
    _logit_cache.clear()
    if features:
        _feature_cache.clear()


def _nodes_and_features(tree: Node):
    """(nodes, features, key) of a tree, as collect_nodes_with_features gives them."""
    key = tree.key()
    features = _feature_cache.get(key)
    if features is None:
        nodes, features = collect_nodes_with_features(tree)
        if len(_feature_cache) >= NODE_CACHE_LIMIT:
            _clear_node_caches()
        _feature_cache[key] = features
        return nodes, features, key
    return collect_nodes_with_parents(tree), features, key


def _node_logits(key, features):
    """The selector's node logits for a tree."""
    if not _logits_repeatable:
        return subtree_selector(features, return_value=True)[0]
    logits = _logit_cache.get(key)
    if logits is None:
        logits = _logit_cache[key] = subtree_selector(features, return_value=True)[0]
    return logits


def set_operator(name, inputs=None) -> None:
    """Choose the crossover operator for the run (see OPERATORS)."""
    global operator_name, semantic_inputs
    if name not in OPERATORS:
        raise ValueError(f"crossover_operator must be one of {OPERATORS}, got {name!r}")
    operator_name, semantic_inputs = name, inputs


def set_selector_device(new_device: torch.device) -> None:
    global subtree_selector, selector_optimizer
    if gp.DEVICE.type == new_device.type:
        return
    gp.DEVICE = new_device
    _clear_node_caches()
    if subtree_selector is not None:
        subtree_selector.to(gp.DEVICE)
    if selector_optimizer is not None:
        for state in selector_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(gp.DEVICE)


def clear_history() -> None:
    """Forget past crossovers and record the feature size of the current operator set."""
    global crossover_history, selector_feature_dim
    # Reset crossover history whenever we start a new GA run so feature sizes stay consistent
    crossover_history = []
    selector_feature_dim = gp.get_selector_feature_dim()
    _clear_node_caches()


def reset_selector(use_learned_crossover, selector_lr, mlp_hidden_dim, use_transformer, transformer_heads,
                   transformer_layers, mlp_layers, mlp_activation, mlp_dropout, use_critic, verbose):
    """Create a fresh selector and optimizer, or none for random crossover."""
    global subtree_selector, selector_optimizer, _logits_repeatable
    _clear_node_caches()
    _logits_repeatable = bool(use_learned_crossover) and not use_transformer and not mlp_dropout > 0
    if use_learned_crossover:
        subtree_selector = SubtreeSelector(
            feature_dim=gp.get_selector_feature_dim(),
            hidden_dim=mlp_hidden_dim,
            use_transformer=use_transformer,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            use_critic=use_critic,  # Pass the critic flag
        )
        selector_optimizer = optim.Adam(subtree_selector.parameters(), lr=selector_lr)
        if verbose:
            critic_mode = "A2C (with critic)" if use_critic else "Policy Gradient (no critic)"
            if use_transformer:
                print(f"Using learned subtree crossover with Transformer encoder ({transformer_layers} layers, {transformer_heads} heads, {critic_mode})")
            else:
                print(f"Using learned subtree crossover with MLP ({mlp_layers} layers, {mlp_hidden_dim} hidden, {mlp_activation} activation, {mlp_dropout} dropout, {critic_mode})")
    else:
        subtree_selector = None
        if verbose:
            print("Using random subtree crossover")


def _structural_bias(pair_logits, a_features, b_features, heuristic_bias_scale):
    """Add the structural heuristic to the joint logits: favour pairs of the same node type
    and operator, and pairs of similar relative size and depth."""
    a_type = torch.argmax(a_features[:, :4], dim=1)
    b_type = torch.argmax(b_features[:, :4], dim=1)
    type_match = (a_type.unsqueeze(1) == b_type.unsqueeze(0)).to(pair_logits.dtype)

    a_size = a_features[:, 4].unsqueeze(1)
    b_size = b_features[:, 4].unsqueeze(0)
    a_depth = a_features[:, 5].unsqueeze(1)
    b_depth = b_features[:, 5].unsqueeze(0)
    size_gap = torch.abs(a_size - b_size)
    depth_gap = torch.abs(a_depth - b_depth)

    a_ops = a_features[:, 10:]
    b_ops = b_features[:, 10:]
    if a_ops.numel() > 0 and b_ops.numel() > 0 and a_ops.shape[1] == b_ops.shape[1]:
        op_match = a_ops @ b_ops.transpose(0, 1)
    else:
        op_match = torch.zeros_like(pair_logits)

    _bs = float(heuristic_bias_scale)
    return pair_logits + _bs * (0.15 * type_match + 0.20 * op_match - 0.10 * size_gap - 0.08 * depth_gap)


def _sample_pair(pair_logits, a_features, b_features, heuristic_bias_scale, temperature):
    """Add the structural bias to the joint logits and sample one (a_idx, b_idx) pair."""
    pair_logits = _structural_bias(pair_logits, a_features, b_features, heuristic_bias_scale)
    safe_temp = max(float(temperature), 1e-6)
    flat_logits = torch.clamp(pair_logits / safe_temp, -10.0, 10.0).reshape(-1)
    probs = torch.softmax(flat_logits, dim=0)
    probs = torch.clamp(probs, min=1e-8)
    probs = probs / probs.sum()

    flat_idx = int(torch.multinomial(probs, 1).item()) if probs.numel() > 1 else 0
    num_b = int(b_features.shape[0])
    a_idx = int(flat_idx // max(num_b, 1))
    b_idx = int(flat_idx % max(num_b, 1))
    return a_idx, b_idx


def _ssc_pick(a_nodes, b_nodes):
    """Semantic similarity-based crossover (Uy et al., 2011): retry uniform point pairs until the two
    subtrees' mean absolute output difference on the training inputs lies inside SSC_BOUNDS, keeping
    the last pair after SSC_TRIALS misses."""
    lower, upper = SSC_BOUNDS
    for _ in range(SSC_TRIALS):
        pa, pb = random.choice(a_nodes), random.choice(b_nodes)
        with np.errstate(all='ignore'):
            dist = np.mean(np.abs(pa[0].evaluate(semantic_inputs) - pb[0].evaluate(semantic_inputs)))
        if lower < dist < upper:
            break
    return pa, pb


def _size_fair_pick(a_nodes, b_nodes):
    """Size-fair crossover (Langdon, 2000), size-limit form: a uniform point in the first parent, then a
    uniform point in the second among subtrees of at most 1 + 2 * (size of the first subtree)."""
    pa = random.choice(a_nodes)
    limit = 1 + 2 * pa[0].size()
    return pa, random.choice([n for n in b_nodes if n[0].size() <= limit])


def learned_subtree_crossover(
    a: Node,
    b: Node,
    use_learned=True,
    temperature=0.5,
    record_gen: int | None = None,
    heuristic_bias_scale: float = 1.0,
) -> Tuple[Node, Node, int | None]:
    """Crossover using learned subtree selection with custom temperature."""
    global subtree_selector, selector_feature_dim, crossover_history

    a_clone = a.clone()
    b_clone = b.clone()
    use_selector = use_learned and subtree_selector is not None

    a_features = None
    b_features = None

    expected_dim = gp.get_selector_feature_dim()
    if use_selector and selector_feature_dim is not None and expected_dim != selector_feature_dim:
        use_selector = False
        crossover_history.clear()

    if use_selector:
        a_nodes, a_features, a_key = _nodes_and_features(a_clone)
        b_nodes, b_features, b_key = _nodes_and_features(b_clone)
        # Use neural network to select a joint subtree pair.
        with torch.no_grad():
            try:
                a_logits = _node_logits(a_key, a_features)
                b_logits = _node_logits(b_key, b_features)

                pair_logits = a_logits.unsqueeze(1) + b_logits.unsqueeze(0)
                a_idx, b_idx = _sample_pair(pair_logits, a_features, b_features, heuristic_bias_scale, temperature)
            except Exception as exc:
                if gp.is_cuda_oom(exc) and gp.DEVICE.type == 'cuda':
                    set_selector_device(torch.device('cpu'))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                a_idx = random.randint(0, len(a_nodes) - 1)
                b_idx = random.randint(0, len(b_nodes) - 1)
        node_a, parent_a, attr_a = a_nodes[a_idx]
        node_b, parent_b, attr_b = b_nodes[b_idx]
    elif use_learned and operator_name == "heuristic":
        # The structural bias alone, no network: the method's pair distribution with zero node logits
        a_nodes, a_feats, _ = _nodes_and_features(a_clone)
        b_nodes, b_feats, _ = _nodes_and_features(b_clone)
        with torch.no_grad():
            zeros = torch.zeros(len(a_nodes), len(b_nodes), dtype=a_feats.dtype, device=a_feats.device)
            a_idx, b_idx = _sample_pair(zeros, a_feats, b_feats, heuristic_bias_scale, temperature)
        node_a, parent_a, attr_a = a_nodes[a_idx]
        node_b, parent_b, attr_b = b_nodes[b_idx]
    elif use_learned and operator_name in ("ssc", "sizefair"):
        pick = _ssc_pick if operator_name == "ssc" else _size_fair_pick
        (node_a, parent_a, attr_a), (node_b, parent_b, attr_b) = pick(
            collect_nodes_with_parents(a_clone), collect_nodes_with_parents(b_clone))
    else:
        # Random selection (baseline)
        a_nodes = collect_nodes_with_parents(a_clone)
        b_nodes = collect_nodes_with_parents(b_clone)
        node_a, parent_a, attr_a = random.choice(a_nodes)
        node_b, parent_b, attr_b = random.choice(b_nodes)

    # Swap subtrees
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

    # Store crossover info for later learning
    history_idx = None
    if use_selector and a_features is not None and b_features is not None:
        crossover_history.append({
            'a_features_all': a_features.cpu(),
            'b_features_all': b_features.cpu(),
            'a_idx': a_idx,
            'b_idx': b_idx,
            'parents': (a.clone(), b.clone()),
            'offspring': (new_a.clone(), new_b.clone()),
            'gen': record_gen,
        })
        history_idx = len(crossover_history) - 1

    return new_a, new_b, history_idx


def update_selector(
    pop: List[Node],
    scores: List[float],
    inputs,
    y,
    complexity_coef=1e-3,
    inclusion_coef=1.0,
    critic_weight=0.5,
    entropy_weight=0.01,
    recent_gen_window=2,
    min_history=64,
    reward_clip=5.0,
    current_gen: int | None = None,
    temperature: float = 0.5,
    heuristic_bias_scale: float = 1.0,
) -> float | None:
    """Update the selector NN. Returns mean raw reward (positive = selector helped), or None."""
    global subtree_selector, selector_optimizer, crossover_history

    _clear_node_caches(features=False)  # the weights may change below
    if not crossover_history or subtree_selector is None:
        return None

    # Filter history to only include records matching the current feature dimension
    expected_dim = gp.get_selector_feature_dim()
    latest_gen = None
    available_gens = []
    for record in crossover_history:
        if 'gen' in record and record['gen'] is not None:
            available_gens.append(record['gen'])

    if available_gens:
        latest_gen = max(available_gens)
        if current_gen is not None:
            desired_gen = current_gen - 1
            if desired_gen in available_gens:
                latest_gen = desired_gen

    min_allowed_gen = None
    if latest_gen is not None:
        min_allowed_gen = latest_gen - max(int(recent_gen_window), 0)

    filtered_history = []
    for record in crossover_history:
        if record['a_features_all'].dim() != 2 or record['b_features_all'].dim() != 2:
            continue
        if record['a_features_all'].size(-1) != expected_dim:
            continue
        if record['b_features_all'].size(-1) != expected_dim:
            continue
        record_gen = record.get('gen')
        if latest_gen is not None and record_gen is None:
            continue
        if min_allowed_gen is not None and record_gen < min_allowed_gen:
            continue
        if latest_gen is not None and record_gen > latest_gen:
            continue
        filtered_history.append(record)
    if not filtered_history:
        crossover_history.clear()
        return None

    if len(filtered_history) < max(1, int(min_history)):
        crossover_history = filtered_history[-max(1, int(min_history) * 4):]
        return None

    # Evaluate offspring fitness (rewards)
    rewards = []

    for record in filtered_history:
        parent1, parent2 = record.get('parents', (None, None))
        child1, child2 = record['offspring']

        if parent1 is not None and parent2 is not None:
            parent_fit1 = fitness(parent1, inputs, y, complexity_coef, inclusion_coef)
            parent_fit2 = fitness(parent2, inputs, y, complexity_coef, inclusion_coef)
            parent_best = min(parent_fit1, parent_fit2)
        else:
            parent_best = sum(scores) / len(scores) if scores else 0.0

        fitness1 = fitness(child1, inputs, y, complexity_coef, inclusion_coef)
        fitness2 = fitness(child2, inputs, y, complexity_coef, inclusion_coef)
        reward = parent_best - min(fitness1, fitness2)
        rewards.append(reward)

    if not rewards:
        crossover_history.clear()
        return None

    mean_reward = float(np.mean(rewards))

    # Convert to tensors
    try:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=gp.DEVICE)

        # Skip update if rewards are all the same or invalid
        rewards_std = rewards_tensor.std(unbiased=False)
        if rewards_std < 1e-8 or torch.isnan(rewards_tensor).any():
            crossover_history.clear()
            return mean_reward

        normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_std + 1e-6)
        if reward_clip is not None and reward_clip > 0:
            normalized_rewards = torch.clamp(normalized_rewards, -float(reward_clip), float(reward_clip))

        selected_log_probs = []
        selected_values = []
        entropies = []

        for record in filtered_history:
            a_features = record['a_features_all'].to(gp.DEVICE)
            b_features = record['b_features_all'].to(gp.DEVICE)
            a_idx = record['a_idx']
            b_idx = record['b_idx']

            action_logits_a, values_a = subtree_selector(a_features, return_value=True)
            action_logits_b, values_b = subtree_selector(b_features, return_value=True)

            pair_logits = action_logits_a.unsqueeze(1) + action_logits_b.unsqueeze(0)
            pair_logits = _structural_bias(pair_logits, a_features, b_features, heuristic_bias_scale)
            safe_temp = max(float(temperature), 1e-6)
            flat_logits = torch.clamp(pair_logits / safe_temp, -10.0, 10.0).reshape(-1)
            pair_log_probs = torch.log_softmax(flat_logits, dim=0)
            pair_probs = torch.softmax(flat_logits, dim=0)

            num_b = int(b_features.shape[0])
            flat_idx = int(a_idx) * max(num_b, 1) + int(b_idx)
            flat_idx = min(max(flat_idx, 0), pair_log_probs.numel() - 1)
            selected_log_probs.append(pair_log_probs[flat_idx])
            entropies.append(-torch.sum(pair_probs * torch.log(pair_probs + 1e-10)))

            if subtree_selector.use_critic and values_a is not None and values_b is not None:
                selected_values.append((values_a[a_idx] + values_b[b_idx]) / 2.0)

        selected_log_probs = torch.stack(selected_log_probs)

        if subtree_selector.use_critic and selected_values:
            selected_values = torch.stack(selected_values)
            advantages = normalized_rewards - selected_values.detach()
            actor_loss = -torch.mean(selected_log_probs * advantages)
            critic_loss = torch.mean((normalized_rewards - selected_values) ** 2)
            critic_weight_local = critic_weight
        else:
            actor_loss = -torch.mean(selected_log_probs * normalized_rewards)
            critic_loss = torch.tensor(0.0, device=gp.DEVICE)
            critic_weight_local = 0.0

        entropy_weight_local = entropy_weight
        entropy_loss = -torch.mean(torch.stack(entropies)) if entropies else torch.tensor(0.0, device=gp.DEVICE)

        total_loss = actor_loss + critic_weight_local * critic_loss + entropy_weight_local * entropy_loss

        # Backpropagation
        if not torch.isnan(total_loss) and not torch.isinf(total_loss):
            selector_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(subtree_selector.parameters(), max_norm=1.0)
            selector_optimizer.step()
    except Exception as exc:
        if gp.is_cuda_oom(exc):
            if gp.DEVICE.type == 'cuda':
                set_selector_device(torch.device('cpu'))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            crossover_history.clear()
            return mean_reward
        raise

    # Clear history
    crossover_history.clear()
    return mean_reward
