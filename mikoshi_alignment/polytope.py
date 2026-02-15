"""SPDP Inference Polytope Geometry.

The mathematical core: shifted partial derivative matrices, inference polytopes,
and admissible region computations.

Reference: geometric safety theory.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from mikoshi_alignment.utils import matrix_rank_approx


def spdp_matrix(
    coefficients: NDArray,
    n_variables: int,
    order: int = 2,
    shift_degree: int = 1,
) -> NDArray:
    """Build the Shifted Partial Derivative (SPDP) matrix.

    Parameters
    ----------
    coefficients : array-like
        Polynomial coefficients (flattened).
    n_variables : int
        Number of variables.
    order : int
        Polynomial order.
    shift_degree : int
        Shift degree for partial derivatives.

    Returns
    -------
    NDArray
        SPDP matrix.
    """
    c = np.asarray(coefficients, dtype=float).ravel()
    # Number of monomials of degree ≤ order in n_variables vars
    # Simplified: use coefficient vector length
    n = len(c)
    size = max(n_variables, n)
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            idx = (i * shift_degree + j) % n
            M[i, j] = c[idx]
    return M


def spdp_rank(matrix: NDArray, projection: Optional[NDArray] = None) -> int:
    """Compute SPDP rank after optional observer projection.

    Parameters
    ----------
    matrix : array-like
        SPDP matrix.
    projection : array-like, optional
        Observer projection matrix.

    Returns
    -------
    int
        Rank.
    """
    M = np.asarray(matrix, dtype=float)
    if projection is not None:
        P = np.asarray(projection, dtype=float)
        M = P @ M if M.shape[0] == P.shape[1] else M
    return matrix_rank_approx(M)


def inference_polytope(constraints: NDArray) -> Dict[str, Any]:
    """Define an admissible region from linear constraints Ax ≤ b.

    Parameters
    ----------
    constraints : array-like
        Shape (m, n+1) where last column is b.

    Returns
    -------
    dict
        Polytope representation with keys ``A``, ``b``, ``n_constraints``, ``dimension``.
    """
    C = np.asarray(constraints, dtype=float)
    A = C[:, :-1]
    b = C[:, -1]
    return {
        "A": A,
        "b": b,
        "n_constraints": A.shape[0],
        "dimension": A.shape[1],
    }


def is_in_polytope(point: NDArray, polytope: Dict[str, Any]) -> bool:
    """Check if *point* is inside the polytope (Ax ≤ b).

    Parameters
    ----------
    point : array-like
        Point to check.
    polytope : dict
        From :func:`inference_polytope`.

    Returns
    -------
    bool
        True if point satisfies all constraints.
    """
    x = np.asarray(point, dtype=float).ravel()
    A = polytope["A"]
    b = polytope["b"]
    return bool(np.all(A @ x <= b + 1e-12))


def polytope_volume(polytope: Dict[str, Any], n_samples: int = 10000, seed: int = 42) -> float:
    """Approximate polytope volume via Monte Carlo sampling.

    Parameters
    ----------
    polytope : dict
        From :func:`inference_polytope`.
    n_samples : int
        Number of random samples.
    seed : int
        Random seed.

    Returns
    -------
    float
        Approximate volume (relative to bounding box).
    """
    A = polytope["A"]
    b = polytope["b"]
    d = polytope["dimension"]
    rng = np.random.default_rng(seed)

    # Estimate bounding box from constraints
    # Use range [-max(|b|), max(|b|)] per dimension
    bound = float(np.max(np.abs(b))) + 1.0
    samples = rng.uniform(-bound, bound, size=(n_samples, d))
    inside = np.sum(np.all(samples @ A.T <= b[None, :] + 1e-12, axis=1))
    box_volume = (2 * bound) ** d
    return float(inside / n_samples * box_volume)


def boundary_distance(point: NDArray, polytope: Dict[str, Any]) -> float:
    """Distance from *point* to nearest polytope boundary.

    Parameters
    ----------
    point : array-like
        Point to check.
    polytope : dict
        From :func:`inference_polytope`.

    Returns
    -------
    float
        Minimum slack.  Negative if outside.
    """
    x = np.asarray(point, dtype=float).ravel()
    A = polytope["A"]
    b = polytope["b"]
    slacks = b - A @ x
    # Normalise by row norms for geometric distance
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    distances = slacks / norms
    return float(np.min(distances))


def visualize_polytope_2d(
    polytope: Dict[str, Any],
    points: Optional[NDArray] = None,
    ax: Any = None,
) -> Any:
    """Visualise a 2-D polytope with matplotlib.

    Parameters
    ----------
    polytope : dict
        Must have dimension 2.
    points : array-like, optional
        Shape (N, 2) — points to overlay.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
        The axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization: pip install matplotlib")

    if polytope["dimension"] != 2:
        raise ValueError("Only 2-D polytopes can be visualized")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    A = polytope["A"]
    b = polytope["b"]
    bound = float(np.max(np.abs(b))) + 1.0

    # Plot feasible region via sampling
    xx = np.linspace(-bound, bound, 200)
    yy = np.linspace(-bound, bound, 200)
    X, Y = np.meshgrid(xx, yy)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.all(grid @ A.T <= b[None, :] + 1e-12, axis=1).reshape(X.shape)
    ax.contourf(X, Y, inside.astype(float), levels=[0.5, 1.5], colors=["#d4edda"], alpha=0.7)
    ax.contour(X, Y, inside.astype(float), levels=[0.5], colors=["green"])

    if points is not None:
        pts = np.asarray(points, dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], c="red", s=20, zorder=5)

    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal")
    ax.set_title("Inference Polytope")
    return ax
