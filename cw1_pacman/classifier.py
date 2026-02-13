"""Tunable MLP classifier for Pacman action prediction.

Interface is kept compatible with classifierAgents.py:
- fit(data, target)
- predict(features, legal)
- reset()
"""

import numpy as np


class Classifier:
    def __init__(self):
        # Model shape
        self.input_dim = 25
        self.hidden_dim = 16
        self.output_dim = 4

        # Training hyperparameters
        self.learning_rate = 0.03
        self.epochs = 250
        self.batch_size = 1  # None => full-batch, 1 => stochastic, int => mini-batch
        self.l2_lambda = 1e-4
        self.momentum = 0.9
        self.lr_decay = 0.997
        self.hidden_activation = "relu"  # relu | tanh | leaky_relu
        self.leaky_slope = 0.01

        # Validation / early stopping
        self.validation_split = 0.2
        self.early_stopping = True
        self.patience = 10
        self.min_delta = 1e-4

        self.seed = 42
        self.rng = np.random.default_rng(self.seed)

        self._is_fitted = False
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.vW1 = None
        self.vb1 = None
        self.vW2 = None
        self.vb2 = None

        self.train_loss_history = []
        self.val_loss_history = []

        # Runtime policy memory (used only at predict-time).
        self.last_action_idx = None
        self.last_feature_key = None
        self.same_state_count = 0
        self.recent_actions = []
        self.recent_feature_keys = []
        self.state_action_counts = {}
        self.stuck_threshold = 3
        self.explore_prob = 0.06
        self.wall_penalty_weight = 1.2
        self.food_bonus_weight = 0.12
        self.ghost_penalty_base = 1.1
        self.ghost_visible_extra = 0.7
        self.reverse_penalty = 0.12
        self.stuck_repeat_penalty = 0.55
        self.stuck_food_bonus = 0.08
        self.hard_safety_on = True
        self.hard_risk_threshold = 1
        self.danger_explore_scale = 0.2
        self.loop_escape_bonus = 0.45
        self.loop_reverse_penalty = 0.45
        self.state_repeat_penalty = 0.35
        self.corridor_penalty = 0.22

    def reset(self):
        self._is_fitted = False
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.vW1 = None
        self.vb1 = None
        self.vW2 = None
        self.vb2 = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.last_action_idx = None
        self.last_feature_key = None
        self.same_state_count = 0
        self.recent_actions = []
        self.recent_feature_keys = []
        self.state_action_counts = {}

    def _initialize_parameters(self):
        if self.hidden_activation == "tanh":
            s1 = np.sqrt(1.0 / self.input_dim)
            s2 = np.sqrt(1.0 / self.hidden_dim)
        else:
            s1 = np.sqrt(2.0 / self.input_dim)
            s2 = np.sqrt(2.0 / self.hidden_dim)

        self.W1 = self.rng.normal(0.0, s1, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = self.rng.normal(0.0, s2, (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((1, self.output_dim))

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def _prepare_X(self, data):
        X = np.asarray(data, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_dim:
            X = np.asarray([self._normalize_feature_vector(row.tolist()) for row in X], dtype=np.float32)
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")
        return X

    def _normalize_feature_vector(self, feature_row):
        """Normalize variable ghost-count feature vectors to fixed length 25."""
        f = [int(v) for v in feature_row]
        if len(f) == self.input_dim:
            return f
        if len(f) < 9:
            return (f + [0] * self.input_dim)[: self.input_dim]

        # api.getFeatureVector structure:
        # walls(4), food(4), ghost blocks(8 each, repeated per ghost), visibleGhost(1)
        walls = f[0:4]
        food = f[4:8]
        visible_ghost = f[-1]
        ghost_flat = f[8:-1]
        ghost_blocks = [ghost_flat[i:i + 8] for i in range(0, len(ghost_flat), 8)]
        ghost_blocks = [b for b in ghost_blocks if len(b) == 8]

        # Keep two strongest ghost blocks and pad if missing.
        ghost_blocks.sort(key=lambda b: sum(b), reverse=True)
        block1 = ghost_blocks[0] if len(ghost_blocks) >= 1 else [0] * 8
        block2 = ghost_blocks[1] if len(ghost_blocks) >= 2 else [0] * 8

        normalized = walls + food + block1 + block2 + [visible_ghost]
        return normalized[: self.input_dim]

    def _prepare_y(self, target):
        y = np.asarray(target, dtype=np.int64).reshape(-1)
        if np.any((y < 0) | (y >= self.output_dim)):
            raise ValueError("Labels must be in [0, 3]")
        return y

    def _one_hot(self, y):
        out = np.zeros((y.shape[0], self.output_dim), dtype=np.float32)
        out[np.arange(y.shape[0]), y] = 1.0
        return out

    def _hidden_activate(self, z):
        if self.hidden_activation == "relu":
            return np.maximum(0.0, z)
        if self.hidden_activation == "tanh":
            return np.tanh(z)
        if self.hidden_activation == "leaky_relu":
            return np.where(z > 0.0, z, self.leaky_slope * z)
        raise ValueError(f"Unsupported activation: {self.hidden_activation}")

    def _hidden_grad(self, z, a):
        if self.hidden_activation == "relu":
            return (z > 0.0).astype(float)
        if self.hidden_activation == "tanh":
            return 1.0 - (a ** 2)
        if self.hidden_activation == "leaky_relu":
            return np.where(z > 0.0, 1.0, self.leaky_slope)
        raise ValueError(f"Unsupported activation: {self.hidden_activation}")

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    def _forward(self, X):
        x_out = X
        y1_net = x_out @ self.W1 + self.b1
        y1_out = self._hidden_activate(y1_net)
        z_net = y1_out @ self.W2 + self.b2
        z_out = self._softmax(z_net)
        cache = {
            "x_out": x_out,
            "y1_net": y1_net,
            "y1_out": y1_out,
            "z_out": z_out,
        }
        return z_out, cache

    def _compute_loss(self, predicted_value, actual_value):
        N = predicted_value.shape[0]
        eps = 1e-12
        ce = -(1.0 / N) * np.sum(actual_value * np.log(predicted_value + eps))
        l2 = 0.5 * self.l2_lambda * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return float(ce + l2)

    def _backward(self, cache, actual_value):
        N = actual_value.shape[0]
        dJ_dz = (cache["z_out"] - actual_value) / N
        dJ_dW2 = cache["y1_out"].T @ dJ_dz + self.l2_lambda * self.W2
        dJ_db2 = np.sum(dJ_dz, axis=0, keepdims=True)

        dJ_dy1_out = dJ_dz @ self.W2.T
        dJ_dy1_net = dJ_dy1_out * self._hidden_grad(cache["y1_net"], cache["y1_out"])
        dJ_dW1 = cache["x_out"].T @ dJ_dy1_net + self.l2_lambda * self.W1
        dJ_db1 = np.sum(dJ_dy1_net, axis=0, keepdims=True)

        return {
            "dJ_dW1": dJ_dW1,
            "dJ_db1": dJ_db1,
            "dJ_dW2": dJ_dW2,
            "dJ_db2": dJ_db2,
        }

    def _step(self, grads):
        self.vW1 = self.momentum * self.vW1 - self.learning_rate * grads["dJ_dW1"]
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * grads["dJ_db1"]
        self.vW2 = self.momentum * self.vW2 - self.learning_rate * grads["dJ_dW2"]
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * grads["dJ_db2"]

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def _split_train_val(self, X, y):
        if self.validation_split <= 0.0:
            return X, y, None, None

        n = X.shape[0]
        n_val = max(1, int(round(n * self.validation_split)))
        idx = self.rng.permutation(n)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if tr_idx.size == 0:
            tr_idx = val_idx
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    def fit(self, data, target):
        X = self._prepare_X(data)
        y = self._prepare_y(target)
        Y = self._one_hot(y)

        X_train, Y_train, X_val, Y_val = self._split_train_val(X, Y)

        self._initialize_parameters()
        self.train_loss_history = []
        self.val_loss_history = []

        best = None
        best_val = float("inf")
        wait = 0

        for _ in range(self.epochs):
            # shuffle training set each epoch
            p = self.rng.permutation(X_train.shape[0])
            Xs = X_train[p]
            Ys = Y_train[p]

            bs = Xs.shape[0] if self.batch_size is None else max(1, int(self.batch_size))
            for start in range(0, Xs.shape[0], bs):
                xb = Xs[start:start + bs]
                yb = Ys[start:start + bs]
                z_out, cache = self._forward(xb)
                grads = self._backward(cache, yb)
                self._step(grads)

            train_pred, _ = self._forward(X_train)
            train_loss = self._compute_loss(train_pred, Y_train)
            self.train_loss_history.append(train_loss)

            if X_val is not None:
                val_pred, _ = self._forward(X_val)
                val_loss = self._compute_loss(val_pred, Y_val)
                self.val_loss_history.append(val_loss)

                if val_loss < (best_val - self.min_delta):
                    best_val = val_loss
                    wait = 0
                    best = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(),
                            self.vW1.copy(), self.vb1.copy(), self.vW2.copy(), self.vb2.copy())
                else:
                    wait += 1
                    if self.early_stopping and wait >= self.patience:
                        break

            self.learning_rate *= self.lr_decay

        if best is not None:
            (self.W1, self.b1, self.W2, self.b2,
             self.vW1, self.vb1, self.vW2, self.vb2) = best

        self._is_fitted = True

    def _filter_illegal(self, probs, legal):
        if legal is None:
            return int(np.argmax(probs))

        legal_to_idx = {"North": 0, "East": 1, "South": 2, "West": 3}
        legal_idx = [legal_to_idx[a] for a in legal if a in legal_to_idx]
        if not legal_idx:
            return int(np.argmax(probs))

        masked = np.full_like(probs, -np.inf, dtype=np.float64)
        masked[legal_idx] = probs[legal_idx]
        return int(np.argmax(masked))

    def _legal_indices(self, legal):
        if legal is None:
            return [0, 1, 2, 3]
        legal_to_idx = {"North": 0, "East": 1, "South": 2, "West": 3}
        idx = [legal_to_idx[a] for a in legal if a in legal_to_idx]
        return idx if idx else [0, 1, 2, 3]

    def _opposite_dir(self, idx):
        # N<->S, E<->W
        return {0: 2, 1: 3, 2: 0, 3: 1}[idx]

    def _policy_adjusted_action(self, feature_vec, probs, legal, state_key):
        """Combine model probability with simple anti-stuck/risk heuristics."""
        legal_idx = self._legal_indices(legal)
        scores = probs.astype(np.float64).copy()

        # Feature layout:
        # walls N,E,S,W | food N,E,S,W | ghost1(8) | ghost2(8) | visibleGhost
        walls = feature_vec[0:4]
        food = feature_vec[4:8]
        ghost1 = feature_vec[8:16]
        ghost2 = feature_vec[16:24]
        visible_ghost = feature_vec[24]

        # N,E,S,W indices inside each 8-neighborhood ghost block.
        dir_to_ghost = {0: 1, 1: 4, 2: 6, 3: 3}
        ghost_risks = {}
        loop_detected = False
        # Detect simple oscillation pattern A,B,A,B.
        if len(self.recent_actions) >= 4:
            a0, a1, a2, a3 = self.recent_actions[-4:]
            if a0 == a2 and a1 == a3 and a0 != a1:
                loop_detected = True
        repeated_window_state = 0
        if self.recent_feature_keys:
            current_key = self.recent_feature_keys[-1]
            repeated_window_state = self.recent_feature_keys.count(current_key)
        feature_loop_detected = False
        if len(self.recent_feature_keys) >= 4:
            k0, k1, k2, k3 = self.recent_feature_keys[-4:]
            if k0 == k2 and k1 == k3 and k0 != k1:
                feature_loop_detected = True

        for d in range(4):
            ghost_risk = max(ghost1[dir_to_ghost[d]], ghost2[dir_to_ghost[d]])
            ghost_risks[d] = ghost_risk
            wall_penalty = self.wall_penalty_weight if walls[d] == 1 else 0.0
            food_bonus = self.food_bonus_weight if food[d] == 1 else 0.0
            danger_penalty = (self.ghost_penalty_base + self.ghost_visible_extra * visible_ghost) * ghost_risk
            repeat_penalty = self.state_repeat_penalty * self.state_action_counts.get((state_key, d), 0)

            scores[d] += food_bonus
            scores[d] -= wall_penalty
            scores[d] -= danger_penalty
            scores[d] -= repeat_penalty

            # Reduce immediate back-and-forth oscillation.
            if self.last_action_idx is not None and d == self._opposite_dir(self.last_action_idx):
                scores[d] -= self.reverse_penalty

            # When a ghost is in front, avoid tunnel-like moves that often trap Pacman.
            if visible_ghost == 1 and d in (0, 2) and walls[1] == 1 and walls[3] == 1:
                scores[d] -= self.corridor_penalty
            if visible_ghost == 1 and d in (1, 3) and walls[0] == 1 and walls[2] == 1:
                scores[d] -= self.corridor_penalty

        # Anti-stuck mode: if local state repeats, discourage repeating same action.
        if self.same_state_count >= self.stuck_threshold and self.last_action_idx is not None:
            scores[self.last_action_idx] -= self.stuck_repeat_penalty
            # Encourage turns that have nearby food when stuck.
            for d in range(4):
                if food[d] == 1:
                    scores[d] += self.stuck_food_bonus
            # Stronger anti-oscillation while stuck.
            opp = self._opposite_dir(self.last_action_idx)
            scores[opp] -= self.loop_reverse_penalty

        if loop_detected:
            # If explicit oscillation found, prefer direction changes.
            if self.last_action_idx is not None:
                scores[self.last_action_idx] -= self.loop_reverse_penalty
                scores[self._opposite_dir(self.last_action_idx)] -= self.loop_reverse_penalty
            for d in range(4):
                if food[d] == 1:
                    scores[d] += self.loop_escape_bonus
        if feature_loop_detected and self.last_action_idx is not None:
            # Break A,B,A,B feature loops even when action loop detection misses.
            scores[self.last_action_idx] -= self.loop_reverse_penalty
            scores[self._opposite_dir(self.last_action_idx)] -= self.loop_reverse_penalty

        # Illegal actions forbidden.
        masked = np.full(4, -np.inf, dtype=np.float64)
        masked[legal_idx] = scores[legal_idx]

        # Hard anti-bounce: avoid immediate reverse unless it is the only viable move.
        if self.last_action_idx is not None and len(legal_idx) > 1:
            opp = self._opposite_dir(self.last_action_idx)
            if opp in legal_idx:
                masked[opp] = -np.inf
                if np.all(np.isneginf(masked[legal_idx])):
                    masked[opp] = scores[opp]

        # Hard safety: if ghost is visible and safe legal moves exist, block risky moves.
        if self.hard_safety_on and visible_ghost == 1:
            safe_idx = [d for d in legal_idx if ghost_risks.get(d, 0) < self.hard_risk_threshold]
            if len(safe_idx) > 0:
                risky_idx = [d for d in legal_idx if d not in safe_idx]
                for d in risky_idx:
                    masked[d] = -np.inf
            else:
                # No fully safe action: pick the least risky legal action.
                best_risk = min(ghost_risks.get(d, 0) for d in legal_idx)
                safer = [d for d in legal_idx if ghost_risks.get(d, 0) == best_risk]
                for d in legal_idx:
                    if d not in safer:
                        masked[d] -= 1.0

        # Forced escape mode if the local state keeps repeating:
        # choose safest legal move, then prefer food, then highest model score.
        if self.same_state_count >= (self.stuck_threshold + 2) or repeated_window_state >= 4 or feature_loop_detected:
            ranked = sorted(
                legal_idx,
                key=lambda d: (
                    -ghost_risks.get(d, 0),   # fewer adjacent ghosts preferred
                    food[d],                  # food direction preferred
                    -self.state_action_counts.get((state_key, d), 0),  # less repeated state-action
                    scores[d],                # model+heuristic score
                ),
                reverse=True,
            )
            # Avoid staying in exact same movement pattern if possible.
            for d in ranked:
                if self.last_action_idx is None:
                    return d
                if d != self.last_action_idx and d != self._opposite_dir(self.last_action_idx):
                    return d
            # If still trapped, cycle deterministically through legal moves.
            if len(legal_idx) > 1 and self.last_action_idx is not None:
                ordered = sorted(legal_idx)
                if self.last_action_idx in ordered:
                    i = ordered.index(self.last_action_idx)
                    return ordered[(i + 1) % len(ordered)]
            return ranked[0]

        # Small exploration (legal only), useful to break long loops.
        effective_explore = self.explore_prob * (self.danger_explore_scale if visible_ghost == 1 else 1.0)
        if loop_detected:
            effective_explore = min(0.25, effective_explore + 0.10)
        if self.rng.random() < effective_explore and len(legal_idx) > 1:
            # Choose best legal alternative different from last action if possible.
            ranked = sorted(legal_idx, key=lambda d: masked[d], reverse=True)
            if self.last_action_idx is not None:
                for d in ranked:
                    if d != self.last_action_idx:
                        return d
            return ranked[0]

        # Directly break A,B,A,B action loops by forcing a third direction.
        if len(self.recent_actions) >= 4:
            a0, a1, a2, a3 = self.recent_actions[-4:]
            if a0 == a2 and a1 == a3 and a0 != a1:
                blocked = {a0, a1}
                alternatives = [d for d in legal_idx if d not in blocked and not np.isneginf(masked[d])]
                if alternatives:
                    return max(alternatives, key=lambda d: masked[d])

        if np.all(np.isneginf(masked[legal_idx])):
            return max(legal_idx, key=lambda d: scores[d])

        return int(np.argmax(masked))

    def predict(self, data, legal=None):
        if not self._is_fitted:
            return 0
        X = self._prepare_X(data)
        z_out, _ = self._forward(X)
        probs = z_out[0]
        feature_vec = X[0]

        # Track repeated local states to detect stuck loops.
        key = tuple(int(v) for v in feature_vec.tolist())
        if key == self.last_feature_key:
            self.same_state_count += 1
        else:
            self.same_state_count = 0
            self.last_feature_key = key
        self.recent_feature_keys.append(key)
        if len(self.recent_feature_keys) > 12:
            self.recent_feature_keys = self.recent_feature_keys[-12:]

        action_idx = self._policy_adjusted_action(feature_vec, probs, legal, key)
        self.state_action_counts[(key, action_idx)] = self.state_action_counts.get((key, action_idx), 0) + 1
        self.last_action_idx = action_idx
        self.recent_actions.append(action_idx)
        if len(self.recent_actions) > 12:
            self.recent_actions = self.recent_actions[-12:]
        return action_idx
