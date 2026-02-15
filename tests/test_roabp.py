"""Tests for roabp_bridge.py."""

import numpy as np
import pytest
from mikoshi_safeguard.roabp_bridge import (
    attention_to_roabp,
    roabp_rank,
    spdp_rank,
    tnn_check_per_head,
    complexity_class_estimate,
    ROABPBridge,
)


class TestAttentionToROABP:
    def test_basic(self):
        W = np.random.rand(2, 4, 4)
        V = np.random.rand(2, 4, 8)
        matrices = attention_to_roabp(W, V)
        assert len(matrices) == 2
        assert matrices[0].shape == (4, 8)

    def test_single_head(self):
        W = np.eye(3).reshape(1, 3, 3)
        V = np.ones((1, 3, 5))
        matrices = attention_to_roabp(W, V)
        assert len(matrices) == 1


class TestROABPRank:
    def test_identity(self):
        M = np.eye(4)
        assert roabp_rank(M, 2) == 2

    def test_zero(self):
        M = np.zeros((4, 4))
        assert roabp_rank(M, 2) == 0

    def test_1d(self):
        assert roabp_rank(np.array([1, 2, 3]), 1) == 1


class TestSPDPRank:
    def test_basic(self):
        P = np.random.rand(4, 4)
        rank = spdp_rank(P)
        assert rank >= 0

    def test_with_projection(self):
        P = np.random.rand(4, 4)
        proj = np.eye(4)
        rank = spdp_rank(P, observer_projection=proj)
        assert rank >= 0


class TestTNNCheckPerHead:
    def test_positive_attention(self):
        W = np.abs(np.random.rand(2, 4, 4))
        V = np.abs(np.random.rand(2, 4, 6))
        results = tnn_check_per_head(W, V)
        assert len(results) == 2
        assert all(isinstance(r, bool) for r in results)


class TestComplexityClassEstimate:
    def test_basic(self):
        W = np.random.rand(3, 8, 8)
        V = np.random.rand(3, 8, 16)
        result = complexity_class_estimate(W, V)
        assert "max_rank" in result
        assert "polynomial_bounded" in result


class TestROABPBridge:
    def test_check(self):
        bridge = ROABPBridge()
        W = np.abs(np.random.rand(2, 4, 4))
        V = np.abs(np.random.rand(2, 4, 8))
        result = bridge.check(W, V)
        assert "score" in result
        assert "passed" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_history(self):
        bridge = ROABPBridge()
        W = np.abs(np.random.rand(2, 4, 4))
        V = np.abs(np.random.rand(2, 4, 8))
        bridge.check(W, V)
        bridge.check(W, V)
        assert len(bridge.history) == 2
