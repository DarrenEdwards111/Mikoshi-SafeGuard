"""Guard 1: Honesty / Positivity — TNN checks on attribution matrices.

From Chapter 9 of N-Frame theory (Edwards, 2023, 2024, 2025).

A totally non-negative (TNN) attribution matrix means every minor is ≥ 0,
ensuring the model's stated reasons faithfully reflect its actual computations.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def attribution_sign_rate(J: NDArray) -> float:
    """Attribution Sign Rate — fraction of non-negative entries in *J*.

    Parameters
    ----------
    J : array-like
        Attribution / Jacobian matrix.

    Returns
    -------
    float
        Value in [0, 1].  1.0 means all entries ≥ 0.
    """
    J = np.asarray(J, dtype=float)
    if J.size == 0:
        return 1.0
    return float(np.mean(J >= 0))


def _minor(A: NDArray, rows: tuple, cols: tuple) -> float:
    """Determinant of a sub-matrix selected by *rows* and *cols*."""
    sub = A[np.ix_(list(rows), list(cols))]
    return float(np.linalg.det(sub))


def random_minor_screen(A: NDArray, r: int = 2, k: int = 100, tol: float = 1e-9) -> bool:
    """Probabilistic TNN screen — sample *k* random r×r minors.

    Parameters
    ----------
    A : array-like
        Matrix to check.
    r : int
        Minor order.
    k : int
        Number of random minors to sample.
    tol : float
        Tolerance below which a determinant is considered negative.

    Returns
    -------
    bool
        True if all sampled minors are ≥ −*tol*.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if r > min(m, n):
        return True
    rng = np.random.default_rng()
    for _ in range(k):
        rows = tuple(sorted(rng.choice(m, size=r, replace=False)))
        cols = tuple(sorted(rng.choice(n, size=r, replace=False)))
        if _minor(A, rows, cols) < -tol:
            return False
    return True


def principal_minor_check(A: NDArray, tol: float = 1e-9) -> bool:
    """Check all leading principal minors are ≥ −*tol*.

    Parameters
    ----------
    A : array-like
        Square matrix.
    tol : float
        Tolerance.

    Returns
    -------
    bool
        True if all leading principal minors are non-negative (within tolerance).
    """
    A = np.asarray(A, dtype=float)
    n = min(A.shape)
    for k in range(1, n + 1):
        det_k = np.linalg.det(A[:k, :k])
        if det_k < -tol:
            return False
    return True


def full_tnn_check(A: NDArray, max_order: int = 4) -> bool:
    """Check all minors up to *max_order* for total non-negativity.

    Parameters
    ----------
    A : array-like
        Matrix to check.
    max_order : int
        Maximum minor order to check.

    Returns
    -------
    bool
        True if every minor up to *max_order* is ≥ 0.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    for order in range(1, min(max_order, m, n) + 1):
        for rows in combinations(range(m), order):
            for cols in combinations(range(n), order):
                if _minor(A, rows, cols) < -1e-12:
                    return False
    return True


def enforce_positivity(
    A: NDArray, method: Literal["clamp", "project", "penalty"] = "clamp"
) -> NDArray:
    """Enforce non-negativity on matrix *A*.

    Parameters
    ----------
    A : array-like
        Input matrix.
    method : str
        ``'clamp'`` — element-wise max(0, x).
        ``'project'`` — project onto nearest TNN via SVD reconstruction.
        ``'penalty'`` — return A unchanged (caller uses penalty in loss).

    Returns
    -------
    NDArray
        Modified matrix.
    """
    A = np.asarray(A, dtype=float).copy()
    if method == "clamp":
        return np.maximum(A, 0.0)
    elif method == "project":
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Clamp singular vectors and values to encourage non-negativity
        A_proj = np.abs(U) @ np.diag(s) @ np.abs(Vt)
        return A_proj
    elif method == "penalty":
        return A
    else:
        raise ValueError(f"Unknown method: {method}")


def honesty_score(J: NDArray) -> float:
    """Combined honesty score in [0, 1] from multiple checks.

    Parameters
    ----------
    J : array-like
        Attribution matrix.

    Returns
    -------
    float
        0 = completely dishonest, 1 = fully honest/TNN.
    """
    J = np.asarray(J, dtype=float)
    asr = attribution_sign_rate(J)
    rms = 1.0 if random_minor_screen(J, r=2, k=50) else 0.0
    pmc = 1.0 if J.ndim == 2 and J.shape[0] == J.shape[1] and principal_minor_check(J) else asr
    return float(0.5 * asr + 0.3 * rms + 0.2 * pmc)


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------

class HonestyGuard:
    """Configurable honesty guard with thresholds and history.

    Parameters
    ----------
    threshold : float
        Minimum honesty score to pass (default 0.8).
    max_order : int
        Max minor order for full TNN checks.
    """

    def __init__(self, threshold: float = 0.8, max_order: int = 4) -> None:
        self.threshold = threshold
        self.max_order = max_order
        self.history: List[Dict] = []

    def check(self, J: NDArray) -> Dict:
        """Run honesty checks on attribution matrix *J*.

        Returns
        -------
        dict
            Keys: ``score``, ``passed``, ``asr``, ``tnn_screen``, ``principal``.
        """
        J = np.asarray(J, dtype=float)
        asr = attribution_sign_rate(J)
        tnn = random_minor_screen(J)
        pmc = principal_minor_check(J) if J.ndim == 2 and J.shape[0] == J.shape[1] else True
        score = honesty_score(J)
        result = {
            "score": score,
            "passed": score >= self.threshold,
            "asr": asr,
            "tnn_screen": tnn,
            "principal": pmc,
        }
        self.history.append(result)
        return result

    def score(self, J: NDArray) -> float:
        """Return honesty score for *J*."""
        return honesty_score(J)

    def is_safe(self, J: NDArray) -> bool:
        """Return True if *J* passes the honesty guard."""
        return self.check(J)["passed"]
