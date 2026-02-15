"""Improvement 4: Representation-Level Monitoring.

Train linear probes on intermediate layers and detect divergence
between external attributions and internal representations.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class LinearProbe:
    """Train a linear probe on intermediate-layer activations.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    """

    def __init__(self, n_classes: int = 2) -> None:
        self.n_classes = n_classes
        self.weights: Optional[NDArray] = None
        self.bias: Optional[NDArray] = None

    def fit(
        self,
        activations: NDArray,
        labels: NDArray,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> "LinearProbe":
        """Train probe via simple gradient descent.

        Parameters
        ----------
        activations : array-like
            Shape (N, D).
        labels : array-like
            Shape (N,) integer labels.
        lr : float
            Learning rate.
        epochs : int
            Training epochs.

        Returns
        -------
        LinearProbe
            Self.
        """
        X = np.asarray(activations, dtype=float)
        y = np.asarray(labels, dtype=int)
        N, D = X.shape
        self.weights = np.zeros((D, self.n_classes))
        self.bias = np.zeros(self.n_classes)

        for _ in range(epochs):
            logits = X @ self.weights + self.bias
            # Softmax
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # One-hot
            targets = np.zeros_like(probs)
            targets[np.arange(N), y] = 1.0
            # Gradient
            grad_logits = (probs - targets) / N
            self.weights -= lr * (X.T @ grad_logits)
            self.bias -= lr * grad_logits.sum(axis=0)

        return self

    def predict(self, activations: NDArray) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        activations : array-like
            Shape (N, D).

        Returns
        -------
        NDArray
            Predicted labels, shape (N,).
        """
        X = np.asarray(activations, dtype=float)
        if self.weights is None:
            raise RuntimeError("Probe not fitted yet")
        logits = X @ self.weights + self.bias
        return np.argmax(logits, axis=1)

    def detect_divergence(
        self,
        external_attributions: NDArray,
        internal_features: NDArray,
    ) -> float:
        """Measure divergence between external attributions and internal features.

        Parameters
        ----------
        external_attributions : array-like
            Shape (N, D) — attributions from the model's output.
        internal_features : array-like
            Shape (N, D) — activations from an intermediate layer.

        Returns
        -------
        float
            Divergence score.  High = internal and external disagree.
        """
        from mikoshi_alignment.utils import cosine_similarity
        ext = np.asarray(external_attributions, dtype=float)
        int_ = np.asarray(internal_features, dtype=float)
        # Compare distributions via mean cosine similarity
        sims = []
        for i in range(min(ext.shape[0], int_.shape[0])):
            sims.append(cosine_similarity(ext[i], int_[i]))
        if not sims:
            return 0.0
        return float(1.0 - np.mean(sims))


def sparse_autoencoder_features(
    activations: NDArray,
    n_features: int = 100,
    sparsity: float = 0.1,
    epochs: int = 50,
    lr: float = 0.01,
) -> NDArray:
    """Extract sparse features via a simple autoencoder.

    Parameters
    ----------
    activations : array-like
        Shape (N, D).
    n_features : int
        Number of sparse features.
    sparsity : float
        L1 penalty weight.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.

    Returns
    -------
    NDArray
        Feature matrix, shape (N, n_features).
    """
    X = np.asarray(activations, dtype=float)
    N, D = X.shape
    rng = np.random.default_rng(42)
    W_enc = rng.standard_normal((D, n_features)) * 0.01
    W_dec = rng.standard_normal((n_features, D)) * 0.01

    for _ in range(epochs):
        # Encode
        hidden = np.maximum(X @ W_enc, 0)  # ReLU
        # Decode
        recon = hidden @ W_dec
        # Loss gradients
        error = recon - X
        grad_dec = hidden.T @ error / N
        grad_hidden = error @ W_dec.T
        grad_hidden[hidden <= 0] = 0  # ReLU grad
        grad_enc = X.T @ grad_hidden / N + sparsity * np.sign(W_enc)

        W_enc -= lr * grad_enc
        W_dec -= lr * grad_dec

    return np.maximum(X @ W_enc, 0)


def internal_external_agreement(
    external_attributions: NDArray,
    internal_activations: NDArray,
) -> float:
    """Score how well surface-level attributions match internal activations.

    Parameters
    ----------
    external_attributions : array-like
        Shape (N, D).
    internal_activations : array-like
        Shape (N, D).

    Returns
    -------
    float
        Agreement score in [0, 1].  1 = perfect agreement.
    """
    probe = LinearProbe(n_classes=2)
    ext = np.asarray(external_attributions, dtype=float)
    int_ = np.asarray(internal_activations, dtype=float)
    divergence = probe.detect_divergence(ext, int_)
    return float(1.0 - divergence)


def mesa_optimization_detector(
    internal_features: NDArray,
    expected_goal_direction: NDArray,
    threshold: float = 0.3,
) -> bool:
    """Flag if internal features suggest a different optimisation goal.

    Parameters
    ----------
    internal_features : array-like
        Shape (N, D) — internal activations.
    expected_goal_direction : array-like
        Shape (D,) — direction of the intended goal in feature space.
    threshold : float
        Alignment threshold.

    Returns
    -------
    bool
        True if mesa-optimisation is suspected.
    """
    from mikoshi_alignment.utils import cosine_similarity
    features = np.asarray(internal_features, dtype=float)
    goal = np.asarray(expected_goal_direction, dtype=float).ravel()
    # Average internal direction
    mean_direction = features.mean(axis=0)
    alignment = cosine_similarity(mean_direction, goal)
    return alignment < threshold


def representation_score(
    external_attributions: NDArray,
    internal_activations: NDArray,
) -> float:
    """Combined representation-level monitoring score.

    Parameters
    ----------
    external_attributions : array-like
        Shape (N, D).
    internal_activations : array-like
        Shape (N, D).

    Returns
    -------
    float
        Score in [0, 1].  1 = representations are consistent.
    """
    return internal_external_agreement(external_attributions, internal_activations)
