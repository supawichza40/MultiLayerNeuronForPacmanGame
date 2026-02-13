"""MLP classifier scaffold for the Pacman coursework.

Notation convention:
- x: input layer values
- x_net: pre-activation at input stage (identity here)
- x_out: output of input stage
- y1_net: hidden layer pre-activation
- y1_out: hidden layer activation output
- z_net: output layer pre-activation
- z_out: output layer activation output (softmax probabilities)
- J: loss value
- dJ_d*: gradient of loss J with respect to *
"""

import numpy as np


class Classifier:
    def __init__(self):
        # Network shape: 25 binary inputs -> hidden layer -> 4 actions.
        self.input_dim = 25
        self.hidden_dim = 16
        self.output_dim = 4

        # Training hyperparameters.
        self.learning_rate = 5e-2
        self.epochs = 300
        self.seed = 42

        self.rng = np.random.default_rng(self.seed)
        self._is_fitted = False

        # Shape convention: row x column.
        # W1: 25 x 16
        # b1: 1 x 16
        # W2: 16 x 4
        # b2: 1 x 4
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
        actual_value = np.zeros((y.shape[0], self.output_dim), dtype=np.float32)
        actual_value[np.arange(y.shape[0]), y] = 1.0
        return actual_value

    def _relu(self, z):
        return np.maximum(0.0, z)

    def _linear(self, x):
        return x

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X):
        x_net = X
        x_out = self._linear(x_net)
        y1_net = np.dot(x_out, self.W1) + self.b1
        y1_out = self._relu(y1_net)
        z_net = np.dot(y1_out, self.W2) + self.b2
        z_out = self._softmax(z_net)

        cache = {
            "x_out": x_out,
            "y1_net": y1_net,
            "y1_out": y1_out,
            "z_net": z_net,
            "z_out": z_out,
        }
        return z_out, cache

    def _compute_loss(self, predicted_value, actual_value):
        N = predicted_value.shape[0]
        eps = 1e-12
        loss = -(1 / N) * np.sum(actual_value * np.log(predicted_value + eps))
        return float(loss)

    def _backward(self, cache, actual_value):
        dJ_dz_net = (cache["z_out"] - actual_value) / actual_value.shape[0]
        dJ_dW2 = np.dot(cache["y1_out"].transpose(), dJ_dz_net)
        dJ_db2 = np.sum(dJ_dz_net, axis=0, keepdims=True)

        dJ_dy1_out = dJ_dz_net @ self.W2.transpose()
        relu_grad = (cache["y1_net"] > 0).astype(float)
        dJ_dy1_net = dJ_dy1_out * relu_grad

        dJ_dW1 = cache["x_out"].transpose() @ dJ_dy1_net
        dJ_db1 = np.sum(dJ_dy1_net, axis=0, keepdims=True)

        grads = {
            "dJ_dW1": dJ_dW1,
            "dJ_db1": dJ_db1,
            "dJ_dW2": dJ_dW2,
            "dJ_db2": dJ_db2,
        }
        return grads

    def _step(self, grads):
        self.W1 -= self.learning_rate * grads["dJ_dW1"]
        self.b1 -= self.learning_rate * grads["dJ_db1"]
        self.W2 -= self.learning_rate * grads["dJ_dW2"]
        self.b2 -= self.learning_rate * grads["dJ_db2"]

    def fit(self, data, target):
        X = self._prepare_X(data)
        y = self._prepare_y(target)
        actual_value = self._one_hot(y)

        self._initialize_parameters()
        self.loss_history = []

        for _epoch in range(self.epochs):
            z_out, cache = self._forward(X)
            J = self._compute_loss(z_out, actual_value)
            grads = self._backward(cache, actual_value)
            self._step(grads)
            self.loss_history.append(J)

        self._is_fitted = True

    def _filter_illegal(self, probs, legal):
        """Mask illegal actions and return best legal action index.

        Label mapping:
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
        if not self._is_fitted:
            return 0

        X = self._prepare_X(data)
        z_out, _cache = self._forward(X)
        p = z_out[0]
        return self._filter_illegal(p, legal)
