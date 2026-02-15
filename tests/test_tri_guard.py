"""Tests for tri_guard.py — Combined Tri-Guard."""

import numpy as np
import pytest
from mikoshi_alignment.tri_guard import TriGuard


class TestTriGuard:
    def test_all_safe(self):
        tg = TriGuard(honesty_threshold=0.5, stability_budget=10.0)
        result = tg.check(
            attribution_matrix=np.ones((3, 3)),
            params=np.array([0.1, 0.1]),
        )
        assert result["safe"] is True
        assert result["score"] > 0

    def test_honesty_fail(self):
        tg = TriGuard(honesty_threshold=0.99, stability_budget=10.0)
        result = tg.check(
            attribution_matrix=-np.ones((3, 3)),
            params=np.array([0.1, 0.1]),
        )
        assert result["honesty"]["passed"] is False

    def test_stability_fail(self):
        tg = TriGuard(stability_budget=0.01)
        result = tg.check(
            attribution_matrix=np.ones((3, 3)),
            params=np.array([10.0, 10.0]),
        )
        assert result["stability"]["passed"] is False

    def test_score_no_check(self):
        tg = TriGuard()
        assert tg.score() == 0.0

    def test_is_safe_no_check(self):
        tg = TriGuard()
        assert tg.is_safe() is False

    def test_report_empty(self):
        tg = TriGuard()
        assert tg.report() == {}

    def test_report_after_check(self):
        tg = TriGuard(stability_budget=10.0)
        tg.check(np.ones((2, 2)), np.array([0.1]))
        report = tg.report()
        assert "honesty" in report
        assert "stability" in report
        assert "holonomy" in report

    def test_history(self):
        tg = TriGuard(stability_budget=10.0)
        tg.check(np.ones((2, 2)), np.array([0.1]))
        tg.check(np.ones((2, 2)), np.array([0.2]))
        assert len(tg.history) == 2

    def test_enforce_safe(self):
        tg = TriGuard(honesty_threshold=0.5, stability_budget=10.0)
        action = {"type": "generate"}
        result = tg.enforce(np.ones((3, 3)), np.array([0.1]), action)
        assert result == action

    def test_enforce_unsafe(self):
        tg = TriGuard(stability_budget=0.001)
        with pytest.raises(RuntimeError, match="Tri-Guard blocked"):
            tg.enforce(np.ones((3, 3)), np.array([10.0, 10.0]), {"type": "bad"})

    def test_with_update_history(self):
        tg = TriGuard(stability_budget=10.0)
        updates = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])
        result = tg.check(np.ones((2, 2)), np.array([0.1]), updates)
        assert "holonomy" in result

    def test_wrap_model(self):
        tg = TriGuard(stability_budget=10.0, honesty_threshold=0.3)
        model = lambda x: x * 2
        wrapped = tg.wrap_model(model)
        result = wrapped(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.array([2.0, 4.0]))
