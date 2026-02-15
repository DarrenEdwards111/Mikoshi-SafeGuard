"""Tests for polytope.py."""

import numpy as np
import pytest
from mikoshi_alignment.polytope import (
    spdp_matrix,
    spdp_rank,
    inference_polytope,
    is_in_polytope,
    polytope_volume,
    boundary_distance,
)


class TestSPDPMatrix:
    def test_basic(self):
        M = spdp_matrix(np.array([1, 2, 3, 4]), n_variables=2)
        assert M.shape[0] >= 2

    def test_single_coeff(self):
        M = spdp_matrix(np.array([1.0]), n_variables=2)
        assert M.shape[0] >= 2


class TestSPDPRank:
    def test_identity(self):
        rank = spdp_rank(np.eye(4))
        assert rank == 4

    def test_zero(self):
        rank = spdp_rank(np.zeros((3, 3)))
        assert rank == 0

    def test_with_projection(self):
        rank = spdp_rank(np.eye(4), projection=np.eye(4))
        assert rank == 4


class TestInferencePolytope:
    def test_basic(self):
        # x1 <= 1, x2 <= 1, -x1 <= 0, -x2 <= 0  →  unit square
        constraints = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=float)
        poly = inference_polytope(constraints)
        assert poly["dimension"] == 2
        assert poly["n_constraints"] == 4


class TestIsInPolytope:
    def test_inside(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        assert is_in_polytope(np.array([0.5, 0.5]), poly) is True

    def test_outside(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        assert is_in_polytope(np.array([2.0, 0.5]), poly) is False

    def test_on_boundary(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        assert is_in_polytope(np.array([1.0, 1.0]), poly) is True


class TestPolytopeVolume:
    def test_unit_square(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        vol = polytope_volume(poly, n_samples=50000)
        assert 0.5 < vol < 2.0  # Should be ~1.0

    def test_positive(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        assert polytope_volume(poly) >= 0


class TestBoundaryDistance:
    def test_interior(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        dist = boundary_distance(np.array([0.5, 0.5]), poly)
        assert dist > 0

    def test_outside(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        dist = boundary_distance(np.array([2.0, 0.5]), poly)
        assert dist < 0

    def test_on_boundary(self):
        constraints = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        poly = inference_polytope(constraints)
        dist = boundary_distance(np.array([1.0, 0.5]), poly)
        assert abs(dist) < 0.01
