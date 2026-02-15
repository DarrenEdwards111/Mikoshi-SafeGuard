"""Tests for representation.py."""

import numpy as np
import pytest
from mikoshi_alignment.representation import (
    LinearProbe,
    sparse_autoencoder_features,
    internal_external_agreement,
    mesa_optimization_detector,
    representation_score,
)


class TestLinearProbe:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        y = (X[:, 0] > 0).astype(int)
        probe = LinearProbe(n_classes=2)
        probe.fit(X, y, epochs=50)
        preds = probe.predict(X)
        assert preds.shape == (50,)
        # Should do better than random
        acc = np.mean(preds == y)
        assert acc > 0.5

    def test_not_fitted(self):
        probe = LinearProbe()
        with pytest.raises(RuntimeError):
            probe.predict(np.ones((5, 3)))

    def test_detect_divergence(self):
        probe = LinearProbe()
        ext = np.ones((10, 5))
        int_ = np.ones((10, 5))
        div = probe.detect_divergence(ext, int_)
        assert div == pytest.approx(0.0)

    def test_divergence_high(self):
        probe = LinearProbe()
        ext = np.ones((10, 5))
        int_ = -np.ones((10, 5))
        div = probe.detect_divergence(ext, int_)
        assert div > 0.5


class TestSparseAutoencoder:
    def test_output_shape(self):
        X = np.random.randn(20, 10)
        features = sparse_autoencoder_features(X, n_features=5, epochs=10)
        assert features.shape == (20, 5)

    def test_non_negative(self):
        X = np.random.randn(20, 10)
        features = sparse_autoencoder_features(X, n_features=5, epochs=10)
        assert np.all(features >= 0)  # ReLU


class TestAgreement:
    def test_identical(self):
        X = np.ones((10, 5))
        score = internal_external_agreement(X, X)
        assert score == pytest.approx(1.0)

    def test_opposite(self):
        score = internal_external_agreement(np.ones((10, 5)), -np.ones((10, 5)))
        assert score < 0.5


class TestMesaOptimization:
    def test_aligned(self):
        features = np.ones((10, 5))
        goal = np.ones(5)
        assert mesa_optimization_detector(features, goal) is False

    def test_misaligned(self):
        features = np.ones((10, 5))
        goal = -np.ones(5)
        assert mesa_optimization_detector(features, goal) is True


class TestRepresentationScore:
    def test_basic(self):
        X = np.random.randn(10, 5)
        score = representation_score(X, X)
        assert 0.0 <= score <= 1.0
