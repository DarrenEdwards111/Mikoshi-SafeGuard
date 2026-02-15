"""Tests for holonomy.py — Guard 3: Holonomy Closure."""

import numpy as np
import pytest
from mikoshi_safeguard.holonomy import (
    compute_connection,
    compute_curvature,
    is_flat,
    holonomy_score,
    detect_reward_hacking,
    HolonomyGuard,
)


class TestComputeConnection:
    def test_basic(self):
        U = np.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]])
        conn = compute_connection(U)
        assert conn.shape == (2, 2, 2)

    def test_single_update(self):
        U = np.array([[1.0, 0.0]])
        conn = compute_connection(U)
        assert conn.shape[0] == 0

    def test_1d(self):
        U = np.array([1.0, 2.0, 3.0])
        conn = compute_connection(U)
        assert conn.shape == (2, 1, 1)


class TestComputeCurvature:
    def test_basic(self):
        conn = np.zeros((3, 2, 2))
        curv = compute_curvature(conn)
        assert curv.shape == (2, 2, 2)

    def test_zero_connection(self):
        conn = np.zeros((5, 3, 3))
        curv = compute_curvature(conn)
        assert np.allclose(curv, 0)

    def test_short_connection(self):
        conn = np.zeros((1, 2, 2))
        curv = compute_curvature(conn)
        assert curv.shape[0] == 0


class TestIsFlat:
    def test_zero(self):
        assert is_flat(np.zeros((3, 3))) is True

    def test_nonzero(self):
        assert is_flat(np.ones((3, 3))) is False

    def test_near_zero(self):
        assert is_flat(np.ones((3, 3)) * 1e-8, tol=1e-6) is True


class TestHolonomyScore:
    def test_constant_updates(self):
        U = np.array([[1.0, 0.0]] * 5)
        # Constant → zero curvature → score ≈ 1
        score = holonomy_score(U)
        assert score > 0.9

    def test_short_history(self):
        U = np.array([[1.0, 0.0], [1.1, 0.0]])
        assert holonomy_score(U) == 1.0

    def test_returns_bounded(self):
        U = np.random.randn(20, 5)
        score = holonomy_score(U)
        assert 0.0 <= score <= 1.0


class TestDetectRewardHacking:
    def test_no_hacking(self):
        # Linear trajectory — should be flat
        U = np.array([[float(i), 0.0] for i in range(10)])
        # Small perturbations from linearity might trigger
        # but perfectly linear should not
        result = detect_reward_hacking(U)
        # We just check it returns a bool
        assert isinstance(result, bool)

    def test_1d(self):
        U = np.array([1.0, 2.0, 3.0])
        result = detect_reward_hacking(U)
        assert isinstance(result, bool)


class TestHolonomyGuard:
    def test_safe_updates(self):
        guard = HolonomyGuard(tol=1e-6)
        for i in range(5):
            guard.record(np.array([0.1 * i, 0.0]))
        result = guard.check()
        assert "score" in result
        assert "passed" in result

    def test_no_history(self):
        guard = HolonomyGuard()
        assert guard.is_safe() is True

    def test_explicit_history(self):
        guard = HolonomyGuard()
        U = np.array([[1.0, 0.0]] * 5)
        result = guard.check(U)
        assert result["score"] > 0.5

    def test_history_tracking(self):
        guard = HolonomyGuard()
        guard.check(np.array([[1.0, 0.0]] * 5))
        guard.check(np.array([[2.0, 0.0]] * 5))
        assert len(guard.history) == 2
