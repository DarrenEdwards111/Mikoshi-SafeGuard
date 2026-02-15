"""Tests for deep_attribution.py."""

import numpy as np
import pytest
from mikoshi_alignment.deep_attribution import (
    integrated_gradients,
    attention_head_decomposition,
    multi_method_attribution,
    attribution_agreement_score,
    layer_wise_relevance,
)


class TestIntegratedGradients:
    def test_linear_model(self):
        weights = np.array([1.0, 2.0, 3.0])
        model = lambda x: np.array([np.dot(weights, x)])
        x = np.array([1.0, 1.0, 1.0])
        ig = integrated_gradients(model, x, steps=20)
        assert ig.shape == x.shape

    def test_with_baseline(self):
        model = lambda x: np.array([np.sum(x ** 2)])
        x = np.array([1.0, 2.0])
        baseline = np.array([0.5, 0.5])
        ig = integrated_gradients(model, x, baseline=baseline, steps=10)
        assert ig.shape == x.shape

    def test_zero_input(self):
        model = lambda x: np.array([np.sum(x)])
        x = np.zeros(3)
        ig = integrated_gradients(model, x, steps=10)
        assert ig.shape == (3,)


class TestAttentionHeadDecomposition:
    def test_basic(self):
        W = np.random.rand(4, 8, 8)  # 4 heads, 8 seq_len
        V = np.random.rand(4, 8, 16)  # 4 heads, 8 seq_len, 16 d_head
        scores = attention_head_decomposition(W, V)
        assert scores.shape == (4,)
        assert np.all(scores >= 0)

    def test_single_head(self):
        W = np.eye(3).reshape(1, 3, 3)
        V = np.ones((1, 3, 5))
        scores = attention_head_decomposition(W, V)
        assert scores.shape == (1,)


class TestMultiMethodAttribution:
    def test_agreement(self):
        attr = {
            "method1": np.array([1, 2, 3]),
            "method2": np.array([1, 2, 3]),
        }
        result = multi_method_attribution(attr)
        assert result["agreement_score"] == pytest.approx(1.0)

    def test_disagreement(self):
        attr = {
            "method1": np.array([1, 0, 0]),
            "method2": np.array([0, 0, 1]),
        }
        result = multi_method_attribution(attr)
        assert result["agreement_score"] < 0.5

    def test_empty(self):
        result = multi_method_attribution({})
        assert result["agreement_score"] == 1.0

    def test_single_method(self):
        result = multi_method_attribution({"a": np.array([1, 2])})
        assert result["agreement_score"] == 1.0


class TestAttributionAgreement:
    def test_identical(self):
        d = {"a": np.ones(5), "b": np.ones(5)}
        assert attribution_agreement_score(d) == pytest.approx(1.0)

    def test_orthogonal(self):
        d = {"a": np.array([1, 0]), "b": np.array([0, 1])}
        assert attribution_agreement_score(d) == pytest.approx(0.0)


class TestLayerWiseRelevance:
    def test_basic(self):
        activations = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0])]
        weights = [np.array([[0.5, 0.3], [0.2, 0.7]]), np.array([[0.6], [0.4]])]
        relevances = layer_wise_relevance(activations, weights, target_index=0)
        assert len(relevances) == 3

    def test_single_layer(self):
        activations = [np.array([1.0])]
        relevances = layer_wise_relevance(activations, [], target_index=0)
        assert len(relevances) == 1
