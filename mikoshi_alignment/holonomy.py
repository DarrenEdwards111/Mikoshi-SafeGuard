"""Guard 3: Holonomy Closure — reward-hacking detection via flat connections.

If the parameter-update connection has non-trivial holonomy around loops in
update space, the model may be cycling through states that game the reward
signal.  A flat connection (zero curvature) guarantees path-independence,
ruling out such exploits.

Reference: N-Frame theory (Edwards, 2023, 2024, 2025).

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def compute_connection(update_sequence: NDArray) -> NDArray:
    """Extract a discrete connection one-form from a sequence of updates.

    The connection A_i is approximated as the log-ratio of consecutive updates.

    Parameters
    ----------
    update_sequence : array-like
        Shape (T, D) — sequence of T parameter-update vectors.

    Returns
    -------
    NDArray
        Connection matrices, shape (T-1, D, D).  Each is a skew-symmetric
        approximation of the transport between consecutive updates.
    """
    U = np.asarray(update_sequence, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    T, D = U.shape
    connections = np.zeros((max(T - 1, 0), D, D))
    for i in range(T - 1):
        diff = U[i + 1] - U[i]
        # Skew-symmetric part: A = (outer - outer^T) / 2
        outer = np.outer(diff, U[i])
        connections[i] = (outer - outer.T) / (np.linalg.norm(U[i]) + 1e-12)
    return connections


def compute_curvature(connection: NDArray) -> NDArray:
    """Compute discrete curvature F = dA + A∧A.

    Parameters
    ----------
    connection : array-like
        Shape (T, D, D) — connection one-forms.

    Returns
    -------
    NDArray
        Curvature two-forms, shape (T-1, D, D).
    """
    A = np.asarray(connection, dtype=float)
    T = A.shape[0]
    if T < 2:
        return np.zeros((0, A.shape[1], A.shape[2]) if A.ndim == 3 else (0,))
    curvature = np.zeros((T - 1, A.shape[1], A.shape[2]))
    for i in range(T - 1):
        dA = A[i + 1] - A[i]
        wedge = A[i] @ A[i + 1] - A[i + 1] @ A[i]
        curvature[i] = dA + wedge
    return curvature


def holonomy_around_loop(connection: NDArray, loop_indices: Sequence[int]) -> NDArray:
    """Compute the holonomy (path-ordered exponential) around a loop.

    Parameters
    ----------
    connection : array-like
        Shape (T, D, D).
    loop_indices : sequence of int
        Indices forming a closed loop.

    Returns
    -------
    NDArray
        Holonomy matrix (D, D).  Identity ⇒ trivial holonomy.
    """
    A = np.asarray(connection, dtype=float)
    D = A.shape[1]
    result = np.eye(D)
    for idx in loop_indices:
        if 0 <= idx < A.shape[0]:
            # Path-ordered: multiply from the left
            from scipy.linalg import expm as _expm  # type: ignore
            result = _expm(A[idx]) @ result
    return result


def _holonomy_around_loop_numpy(connection: NDArray, loop_indices: Sequence[int]) -> NDArray:
    """Fallback holonomy using first-order Taylor (no scipy)."""
    A = np.asarray(connection, dtype=float)
    D = A.shape[1]
    result = np.eye(D)
    for idx in loop_indices:
        if 0 <= idx < A.shape[0]:
            result = (np.eye(D) + A[idx]) @ result
    return result


def is_flat(curvature: NDArray, tol: float = 1e-6) -> bool:
    """Check if curvature is zero (flat connection).

    Parameters
    ----------
    curvature : array-like
        Curvature tensor(s).
    tol : float
        Tolerance for Frobenius norm.

    Returns
    -------
    bool
        True if connection is flat.
    """
    F = np.asarray(curvature, dtype=float)
    return bool(np.linalg.norm(F) < tol)


def detect_reward_hacking(
    update_history: NDArray,
    loop_generators: Optional[List[List[int]]] = None,
) -> bool:
    """Check all generating loops for non-trivial holonomy.

    Parameters
    ----------
    update_history : array-like
        Shape (T, D).
    loop_generators : list of lists of int, optional
        Loops to check.  Defaults to overlapping windows.

    Returns
    -------
    bool
        True if reward hacking is detected.
    """
    U = np.asarray(update_history, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    conn = compute_connection(U)
    curv = compute_curvature(conn)

    if not is_flat(curv, tol=1e-4):
        return True

    if loop_generators:
        for loop in loop_generators:
            try:
                hol = holonomy_around_loop(conn, loop)
            except ImportError:
                hol = _holonomy_around_loop_numpy(conn, loop)
            D = hol.shape[0]
            if np.linalg.norm(hol - np.eye(D)) > 1e-4:
                return True
    return False


def holonomy_score(update_history: NDArray) -> float:
    """Combined holonomy score in [0, 1].  1.0 = flat (safe).

    Parameters
    ----------
    update_history : array-like
        Shape (T, D).

    Returns
    -------
    float
        Score in [0, 1].
    """
    U = np.asarray(update_history, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if U.shape[0] < 3:
        return 1.0
    conn = compute_connection(U)
    curv = compute_curvature(conn)
    norm = float(np.linalg.norm(curv))
    # Sigmoid-like mapping: high curvature → low score
    return float(1.0 / (1.0 + norm))


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------

class HolonomyGuard:
    """Tracks updates and checks for cyclic reward-hacking exploits.

    Parameters
    ----------
    tol : float
        Flatness tolerance.
    window : int
        Number of recent updates to consider.
    """

    def __init__(self, tol: float = 1e-6, window: int = 50) -> None:
        self.tol = tol
        self.window = window
        self.updates: List[NDArray] = []
        self.history: List[Dict] = []

    def record(self, update: NDArray) -> None:
        """Record a parameter update."""
        self.updates.append(np.asarray(update, dtype=float))

    def check(self, update_history: Optional[NDArray] = None) -> Dict:
        """Run holonomy check.

        Parameters
        ----------
        update_history : array-like, optional
            Override internal history.

        Returns
        -------
        dict
            Keys: ``score``, ``passed``, ``hacking_detected``.
        """
        if update_history is not None:
            U = np.asarray(update_history, dtype=float)
        else:
            if len(self.updates) < 3:
                result = {"score": 1.0, "passed": True, "hacking_detected": False}
                self.history.append(result)
                return result
            U = np.array(self.updates[-self.window :])

        score = holonomy_score(U)
        hacking = detect_reward_hacking(U)
        result = {
            "score": score,
            "passed": score >= 0.5 and not hacking,
            "hacking_detected": hacking,
        }
        self.history.append(result)
        return result

    def score(self, update_history: Optional[NDArray] = None) -> float:
        """Return holonomy score."""
        if update_history is not None:
            return holonomy_score(update_history)
        if len(self.updates) < 3:
            return 1.0
        return holonomy_score(np.array(self.updates[-self.window :]))

    def is_safe(self, update_history: Optional[NDArray] = None) -> bool:
        """Return True if no reward hacking detected."""
        return self.check(update_history)["passed"]
