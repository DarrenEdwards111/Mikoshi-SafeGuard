"""Tests for honesty.py — Guard 1: Honesty/Positivity."""

import numpy as np
import pytest
from mikoshi_safeguard.honesty import (
    attribution_sign_rate,
    random_minor_screen,
    principal_minor_check,
    full_tnn_check,
    enforce_positivity,
    honesty_score,
    HonestyGuard,
)


class TestAttributionSignRate:
    def test_all_positive(self):
        assert attribution_sign_rate(np.ones((3, 3))) == 1.0

    def test_all_negative(self):
        assert attribution_sign_rate(-np.ones((3, 3))) == 0.0

    def test_mixed(self):
        J = np.array([[1, -1], [1, 1]])
        assert attribution_sign_rate(J) == 0.75

    def test_empty(self):
        assert attribution_sign_rate(np.array([])) == 1.0

    def test_zeros(self):
        assert attribution_sign_rate(np.zeros((2, 2))) == 1.0

    def test_single_element_positive(self):
        assert attribution_sign_rate(np.array([[5.0]])) == 1.0

    def test_single_element_negative(self):
        assert attribution_sign_rate(np.array([[-1.0]])) == 0.0


class TestRandomMinorScreen:
    def test_positive_matrix(self):
        A = np.array([[1, 2], [3, 4]])
        # det = -2 < 0, not TNN
        # But random_minor_screen checks r=2 minors
        # For 2x2, the only 2x2 minor is det(A) = -2
        assert random_minor_screen(A, r=2, k=10) is False

    def test_identity(self):
        assert random_minor_screen(np.eye(3), r=2, k=50) is True

    def test_tnn_matrix(self):
        # A known TNN matrix
        A = np.array([[1, 1], [0, 1]])
        assert random_minor_screen(A, r=2, k=50) is True

    def test_small_matrix(self):
        # r > min dimension
        A = np.array([[1.0]])
        assert random_minor_screen(A, r=2) is True


class TestPrincipalMinorCheck:
    def test_positive_definite(self):
        A = np.array([[2, 1], [1, 2]])
        assert principal_minor_check(A) is True

    def test_negative_definite(self):
        A = np.array([[-2, 0], [0, -2]])
        assert principal_minor_check(A) is False

    def test_identity(self):
        assert principal_minor_check(np.eye(4)) is True

    def test_singular(self):
        A = np.array([[0, 0], [0, 0]])
        assert principal_minor_check(A) is True  # all minors are 0


class TestFullTnnCheck:
    def test_identity(self):
        assert full_tnn_check(np.eye(3), max_order=3) is True

    def test_negative_entry(self):
        A = np.array([[1, -1], [0, 1]])
        # 1x1 minor at (0,1) = -1 < 0
        assert full_tnn_check(A, max_order=1) is False

    def test_tnn_matrix(self):
        A = np.array([[1, 1], [0, 1]])
        assert full_tnn_check(A, max_order=2) is True


class TestEnforcePositivity:
    def test_clamp(self):
        A = np.array([[-1, 2], [-3, 4]])
        result = enforce_positivity(A, method="clamp")
        assert np.all(result >= 0)
        assert result[0, 1] == 2.0

    def test_project(self):
        A = np.array([[-1, 2], [-3, 4]])
        result = enforce_positivity(A, method="project")
        assert result.shape == A.shape

    def test_penalty(self):
        A = np.array([[-1, 2], [-3, 4]])
        result = enforce_positivity(A, method="penalty")
        np.testing.assert_array_equal(result, A)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            enforce_positivity(np.eye(2), method="invalid")


class TestHonestyScore:
    def test_positive_matrix(self):
        J = np.ones((3, 3))
        score = honesty_score(J)
        assert 0.0 <= score <= 1.0

    def test_negative_matrix(self):
        J = -np.ones((3, 3))
        score = honesty_score(J)
        assert score < 0.5

    def test_identity(self):
        score = honesty_score(np.eye(3))
        assert score > 0.5


class TestHonestyGuard:
    def test_safe_matrix(self):
        guard = HonestyGuard(threshold=0.5)
        J = np.ones((3, 3))
        assert guard.is_safe(J) is True

    def test_unsafe_matrix(self):
        guard = HonestyGuard(threshold=0.9)
        J = -np.ones((3, 3))
        assert guard.is_safe(J) is False

    def test_history(self):
        guard = HonestyGuard()
        guard.check(np.eye(3))
        guard.check(-np.eye(3))
        assert len(guard.history) == 2

    def test_score_method(self):
        guard = HonestyGuard()
        score = guard.score(np.ones((2, 2)))
        assert 0.0 <= score <= 1.0
