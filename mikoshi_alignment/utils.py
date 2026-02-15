"""Shared utilities for Mikoshi AI Alignment.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : array-like
        Input vectors (1-D).

    Returns
    -------
    float
        Cosine similarity in [-1, 1].  Returns 0.0 if either vector is zero.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def matrix_rank_approx(M: NDArray, tol: float = 1e-6) -> int:
    """Approximate numerical rank of a matrix via SVD.

    Parameters
    ----------
    M : array-like
        Input matrix.
    tol : float
        Singular values below *tol* are treated as zero.

    Returns
    -------
    int
        Numerical rank.
    """
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > tol))


def log_determinant(M: NDArray) -> float:
    """Compute log|det(M)| safely via LU decomposition.

    Parameters
    ----------
    M : array-like
        Square matrix.

    Returns
    -------
    float
        Natural log of the absolute determinant.  Returns -inf for singular matrices.
    """
    M = np.asarray(M, dtype=float)
    sign, logdet = np.linalg.slogdet(M)
    if sign == 0:
        return float("-inf")
    return float(logdet)


def numerical_gradient(f: Callable[[NDArray], float], x: NDArray, eps: float = 1e-5) -> NDArray:
    """Compute numerical gradient of scalar function *f* at *x* using central differences.

    Parameters
    ----------
    f : callable
        Scalar-valued function accepting a 1-D array.
    x : array-like
        Point at which to evaluate the gradient.
    eps : float
        Step size for finite differences.

    Returns
    -------
    NDArray
        Gradient vector, same shape as *x*.
    """
    x = np.asarray(x, dtype=float).copy()
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus.flat[i] += eps
        x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def safe_json_log(data: Any, path: str) -> None:
    """Append *data* as a JSON line to file at *path*.

    Creates parent directories if needed.

    Parameters
    ----------
    data : Any
        JSON-serialisable object.
    path : str
        File path for the log.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "a") as fh:
        fh.write(json.dumps(data, default=_default) + "\n")
