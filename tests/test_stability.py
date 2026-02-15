"""Tests for stability.py — Guard 2: Wall Stability."""

import numpy as np
import pytest
from mikoshi_alignment.stability import (
    capability_energy,
    safety_tension,
    barrier_lyapunov,
    curvature_check,
    israel_junction_check,
    stability_score,
    WallStabilityGuard,
)


class TestCapabilityEnergy:
    def test_l2(self):
        assert capability_energy(np.array([3, 4]), "l2") == pytest.approx(5.0)

    def test_l1(self):
        assert capability_energy(np.array([3, -4]), "l1") == pytest.approx(7.0)

    def test_linf(self):
        assert capability_energy(np.array([3, -4]), "linf") == pytest.approx(4.0)

    def test_zero(self):
        assert capability_energy(np.zeros(5)) == 0.0

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            capability_energy(np.ones(3), "l3")


class TestSafetyTension:
    def test_safe(self):
        assert safety_tension(10.0, 3.0) == 7.0

    def test_breach(self):
        assert safety_tension(5.0, 8.0) == -3.0

    def test_exact(self):
        assert safety_tension(5.0, 5.0) == 0.0


class TestBarrierLyapunov:
    def test_safe(self):
        val = barrier_lyapunov(0.5, 1.0)
        assert np.isfinite(val)

    def test_breach(self):
        val = barrier_lyapunov(1.0, 1.0)
        assert val == float("inf")

    def test_over_budget(self):
        val = barrier_lyapunov(1.5, 1.0)
        assert val == float("inf")

    def test_zero_energy(self):
        val = barrier_lyapunov(0.0, 1.0)
        assert np.isfinite(val)


class TestCurvatureCheck:
    def test_stable_trajectory(self):
        traj = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
        assert curvature_check(traj, budget=10.0) is True

    def test_short_trajectory(self):
        traj = np.array([[0.1, 0.1], [0.2, 0.2]])
        assert curvature_check(traj, budget=1.0) is True

    def test_1d(self):
        traj = np.array([0.1, 0.2, 0.3])
        assert curvature_check(traj, budget=10.0) is True


class TestIsraelJunction:
    def test_stable(self):
        assert israel_junction_check(1.0, 1.5, 1.0) is True

    def test_unstable(self):
        assert israel_junction_check(1.0, 5.0, 1.0) is False

    def test_equal_curvature(self):
        assert israel_junction_check(2.0, 2.0, 0.0) is True


class TestStabilityScore:
    def test_safe(self):
        score = stability_score(np.array([0.1, 0.1]), budget=1.0)
        assert 0.0 < score <= 1.0

    def test_breach(self):
        score = stability_score(np.array([10.0, 10.0]), budget=1.0)
        assert score == 0.0

    def test_with_trajectory(self):
        params = np.array([0.1, 0.1])
        traj = np.array([[0.05, 0.05], [0.08, 0.08], [0.1, 0.1]])
        score = stability_score(params, budget=1.0, trajectory=traj)
        assert 0.0 <= score <= 1.0


class TestWallStabilityGuard:
    def test_safe(self):
        guard = WallStabilityGuard(budget=10.0)
        assert guard.is_safe(np.array([0.1, 0.1])) is True

    def test_unsafe(self):
        guard = WallStabilityGuard(budget=0.1)
        assert guard.is_safe(np.array([10.0, 10.0])) is False

    def test_history(self):
        guard = WallStabilityGuard(budget=5.0)
        guard.check(np.array([0.1]))
        guard.check(np.array([0.2]))
        assert len(guard.history) == 2

    def test_trajectory_tracking(self):
        guard = WallStabilityGuard(budget=10.0)
        for i in range(5):
            guard.check(np.array([0.1 * i]))
        assert len(guard.trajectory) == 5
