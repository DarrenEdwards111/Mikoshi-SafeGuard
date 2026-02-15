"""Tests for drift.py."""

import numpy as np
import pytest
from mikoshi_alignment.drift import (
    exponential_moving_drift,
    changepoint_detection,
    DriftDetector,
)


class TestExponentialMovingDrift:
    def test_constant(self):
        H = np.array([[1.0, 0.0]] * 5)
        ema = exponential_moving_drift(H)
        assert np.allclose(ema, 0.0)

    def test_linear(self):
        H = np.array([[float(i), 0.0] for i in range(10)])
        ema = exponential_moving_drift(H, alpha=0.5)
        assert len(ema) == 9
        assert all(np.isfinite(ema))

    def test_single_point(self):
        H = np.array([[1.0, 2.0]])
        ema = exponential_moving_drift(H)
        assert len(ema) == 1
        assert ema[0] == 0.0


class TestChangepointDetection:
    def test_no_change(self):
        signal = np.ones(20)
        cp = changepoint_detection(signal, method="threshold", threshold=0.5)
        assert len(cp) == 0

    def test_step_change(self):
        signal = np.concatenate([np.zeros(10), np.ones(10) * 5])
        cp = changepoint_detection(signal, method="threshold", threshold=2.0)
        assert 10 in cp

    def test_cusum(self):
        signal = np.concatenate([np.zeros(10), np.ones(10) * 3])
        cp = changepoint_detection(signal, method="cusum", threshold=5.0)
        assert isinstance(cp, list)


class TestDriftDetector:
    def test_record(self):
        dd = DriftDetector()
        dd.record(0, np.array([1.0, 2.0]))
        dd.record(1, np.array([1.1, 2.1]))
        assert len(dd.params_history) == 2

    def test_cumulative_drift(self):
        dd = DriftDetector()
        for i in range(10):
            dd.record(i, np.array([0.1 * i, 0.0]))
        drift = dd.cumulative_drift()
        assert drift > 0

    def test_no_drift(self):
        dd = DriftDetector()
        for i in range(5):
            dd.record(i, np.array([1.0, 1.0]))
        assert dd.cumulative_drift() == pytest.approx(0.0)

    def test_drift_score_no_data(self):
        dd = DriftDetector()
        assert dd.drift_score() == 1.0

    def test_drift_score_with_data(self):
        dd = DriftDetector()
        for i in range(20):
            dd.record(i, np.array([0.01 * i]))
        score = dd.drift_score()
        assert 0.0 <= score <= 1.0

    def test_boiling_frog_false(self):
        dd = DriftDetector(threshold=0.1)
        dd.record(0, np.array([0.0]))
        dd.record(1, np.array([0.0]))
        assert dd.detect_boiling_frog() is False

    def test_safe_region_shift(self):
        dd = DriftDetector()
        dist = dd.safe_region_shift(np.array([0, 0]), np.array([3, 4]))
        assert dist == pytest.approx(5.0)
