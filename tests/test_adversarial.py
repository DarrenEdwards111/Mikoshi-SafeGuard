"""Tests for adversarial.py."""

import numpy as np
import pytest
from mikoshi_alignment.adversarial import (
    generate_adversarial_inputs,
    fuzz_attribution,
    guard_robustness_score,
    StressTest,
    find_guard_gaps,
)


class TestGenerateAdversarial:
    def test_produces_outputs(self):
        check = lambda x: bool(np.linalg.norm(x) < 1.0)
        inputs = generate_adversarial_inputs(check, (3,), n=10, seed=42)
        assert len(inputs) == 10
        assert all(x.shape == (3,) for x in inputs)

    def test_always_pass(self):
        check = lambda x: True
        inputs = generate_adversarial_inputs(check, (2,), n=5, seed=0)
        assert len(inputs) == 5


class TestFuzzAttribution:
    def test_basic(self):
        A = np.array([[0.01, 0.01], [0.01, 0.01]])
        results = fuzz_attribution(A, epsilon=0.1, n=10, seed=42)
        assert len(results) == 10
        for arr, flipped in results:
            assert arr.shape == A.shape
            assert isinstance(flipped, bool)

    def test_large_values_no_flip(self):
        A = np.ones((3, 3)) * 100
        results = fuzz_attribution(A, epsilon=0.01, n=10, seed=42)
        flips = sum(1 for _, f in results if f)
        assert flips == 0

    def test_zero_matrix(self):
        A = np.zeros((2, 2))
        results = fuzz_attribution(A, epsilon=0.01, n=5, seed=42)
        assert len(results) == 5


class TestGuardRobustness:
    def test_always_true(self):
        score = guard_robustness_score(lambda x: True, [np.ones(3)] * 5)
        assert score == 1.0

    def test_empty_suite(self):
        assert guard_robustness_score(lambda x: True, []) == 1.0


class TestStressTest:
    def test_run(self):
        guards = {"always_pass": lambda x: True}
        st = StressTest(guards)
        scores = st.run((3,), n_inputs=10, seed=42)
        assert "always_pass" in scores
        assert scores["always_pass"] == 1.0

    def test_report(self):
        guards = {"g1": lambda x: True}
        st = StressTest(guards)
        st.run((2,), n_inputs=5, seed=0)
        report = st.report()
        assert "g1" in report


class TestFindGuardGaps:
    def test_all_pass(self):
        checks = {"g1": lambda x: True, "g2": lambda x: True}
        gaps = find_guard_gaps(checks, (3,), n=20, seed=42)
        assert len(gaps) == 20

    def test_none_pass(self):
        checks = {"g1": lambda x: False}
        gaps = find_guard_gaps(checks, (3,), n=20, seed=42)
        assert len(gaps) == 0
