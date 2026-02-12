"""MLP classifier scaffold for the Pacman coursework.

This file intentionally leaves core learning math as TODOs so you can
implement and understand the full pipeline yourself.
"""

import numpy as np


class Classifier:
    def __init__(self):
        # Network shape based on coursework data:
        # 25 binary inputs -> hidden layer -> 4 action classes.
        self.input_dim = 25
        self.hidden_dim = 12
        self.output_dim = 4

        # Training hyperparameters (adjust as you experiment).
        self.learning_rate = 1e-3
        self.epochs = 100
        self.seed = 42

        self.rng = np.random.default_rng(self.seed)
        self._is_fitted = False

        # Parameters + optional training history.
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss_history = []

    def reset(self):
        """Called by the agent when an episode ends."""
        self._is_fitted = False
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss_history = []

    def _initialize_parameters(self):
        """Initialize trainable parameters for a 1-hidden-layer MLP."""
        # He-style init is a good default with ReLU.
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((1, self.output_dim))

    def _prepare_X(self, data):
        """Convert raw feature rows to float numpy array with shape (N, 25)."""
        X = np.asarray(data, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")
        return X

    def _prepare_y(self, target):
        """Convert labels to int numpy array with shape (N,)."""
        y = np.asarray(target, dtype=np.int64).reshape(-1)
        if np.any((y < 0) | (y >= self.output_dim)):
            raise ValueError("Labels must be in [0, 3]")
        return y

    def _one_hot(self, y):
        """One-hot encode labels for cross-entropy training."""
        y_one_hot = np.zeros((y.shape[0], self.output_dim), dtype=np.float32)
        y_one_hot[np.arange(y.shape[0]), y] = 1.0
        return y_one_hot

    def _relu(self, z):
        return np.maximum(0.0, z)

    def _softmax(self, z):
        # Numerically stable softmax.
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X):
        """Forward pass.

        TODO (you):
        1) Compute hidden pre-activation z1.
        2) Apply ReLU to get a1.
        3) Compute output pre-activation z2.
        4) Apply softmax to get class probabilities probs.
        5) Return probs and a cache needed for backward pass.
        """
        # Example cache keys you may want:
        # cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probs}
        raise NotImplementedError("Implement _forward")

    def _compute_loss(self, probs, y_one_hot):
        """Cross-entropy loss.

        TODO (you):
        - Implement mean cross-entropy:
          L = -(1/N) * sum(y_one_hot * log(probs + eps))
        """
        raise NotImplementedError("Implement _compute_loss")

    def _backward(self, cache, y_one_hot):
        """Backward pass.

        TODO (you):
        1) Compute dL/dz2 from softmax + cross-entropy.
        2) Compute gradients for W2, b2.
        3) Backprop through ReLU for hidden layer.
        4) Compute gradients for W1, b1.
        5) Return grads dict with keys: dW1, db1, dW2, db2.
        """
        raise NotImplementedError("Implement _backward")

    def _step(self, grads):
        """Parameter update step.

        TODO (you):
        - Apply gradient descent update using self.learning_rate.
        """
        raise NotImplementedError("Implement _step")

    def fit(self, data, target):
        """Train model on coursework samples."""
        X = self._prepare_X(data)
        y = self._prepare_y(target)
        y_one_hot = self._one_hot(y)

        self._initialize_parameters()
        self.loss_history = []

        for _epoch in range(self.epochs):
            # TODO (you): full train loop
            # 1) probs, cache = self._forward(X)
            # 2) loss = self._compute_loss(probs, y_one_hot)
            # 3) grads = self._backward(cache, y_one_hot)
            # 4) self._step(grads)
            # 5) self.loss_history.append(loss)
            pass

        self._is_fitted = True

    def _filter_illegal(self, probs, legal):
        """Mask illegal actions and return best legal action index.

        Label mapping in classifierAgents.py:
        0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
        """
        if legal is None:
            return int(np.argmax(probs))

        legal_to_idx = {"North": 0, "East": 1, "South": 2, "West": 3}
        legal_idx = [legal_to_idx[a] for a in legal if a in legal_to_idx]

        if not legal_idx:
            return int(np.argmax(probs))

        masked = np.full_like(probs, -np.inf, dtype=np.float64)
        masked[legal_idx] = probs[legal_idx]
        return int(np.argmax(masked))

    def predict(self, data, legal=None):
        """Predict action class index in [0, 3]."""
        # Safe fallback until model is implemented/trained.
        if not self._is_fitted:
            return 0

        X = self._prepare_X(data)

        # TODO (you): run forward pass and take first row probabilities.
        # probs, _cache = self._forward(X)
        # p = probs[0]
        # return self._filter_illegal(p, legal)
        return 0
