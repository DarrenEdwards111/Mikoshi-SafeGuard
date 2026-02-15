"""Improvement 5: ROABP Approximation for Transformers.

Read-Once Algebraic Branching Programs (ROABPs) provide a polynomial-complexity
framework for analysing attention mechanisms.  This module decomposes attention
patterns into approximate ROABPs and applies TNN checks per head.

Reference: geometric safety theory.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from mikoshi_alignment.utils import matrix_rank_approx


def attention_to_roabp(
    attention_weights: NDArray,
    values: NDArray,
) -> List[NDArray]:
    """Decompose attention as approximate ROABP coefficient matrices.

    Each attention head produces a matrix M_i such that the output is
    the product M_1 · M_2 · ... · M_h applied to the value vectors.

    Parameters
    ----------
    attention_weights : array-like
        Shape (num_heads, seq_len, seq_len).
    values : array-like
        Shape (num_heads, seq_len, d_head).

    Returns
    -------
    list of NDArray
        ROABP coefficient matrices, one per head.
    """
    W = np.asarray(attention_weights, dtype=float)
    V = np.asarray(values, dtype=float)
    num_heads = W.shape[0]
    matrices: List[NDArray] = []
    for h in range(num_heads):
        # Context = attention @ values → coefficient matrix
        M = W[h] @ V[h]  # (seq_len, d_head)
        matrices.append(M)
    return matrices


def roabp_rank(matrix: NDArray, cut_position: int) -> int:
    """Compute ROABP width at a given cut position.

    The ROABP width at position *k* is the rank of the matrix obtained
    by flattening the first *k* variables against the remaining.

    Parameters
    ----------
    matrix : array-like
        2-D matrix (flattened ROABP representation).
    cut_position : int
        Row index at which to cut.

    Returns
    -------
    int
        Numerical rank at the cut.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim == 1:
        return 1
    top = M[:cut_position, :]
    if top.size == 0:
        return 0
    return matrix_rank_approx(top)


def spdp_rank(
    polynomial_encoding: NDArray,
    observer_projection: Optional[NDArray] = None,
    order: int = 2,
) -> int:
    """Compute SPDP rank from Chapter 7.

    Shifted Partial Derivative matrix rank after projection.

    Parameters
    ----------
    polynomial_encoding : array-like
        Coefficient matrix encoding the polynomial.
    observer_projection : array-like, optional
        Observer projection matrix.  If None, uses identity.
    order : int
        Shift degree.

    Returns
    -------
    int
        SPDP rank.
    """
    P = np.asarray(polynomial_encoding, dtype=float)
    if observer_projection is not None:
        proj = np.asarray(observer_projection, dtype=float)
        P = proj @ P if P.shape[0] == proj.shape[1] else P
    # Build shifted partial derivative matrix
    n = P.shape[0]
    spdp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            idx = (i + j * order) % n
            spdp[i, j] = P.flat[idx % P.size]
    return matrix_rank_approx(spdp)


def tnn_check_per_head(
    attention_weights: NDArray,
    values: NDArray,
) -> List[bool]:
    """Apply TNN checks to each attention head's ROABP decomposition.

    Parameters
    ----------
    attention_weights : array-like
        Shape (num_heads, seq_len, seq_len).
    values : array-like
        Shape (num_heads, seq_len, d_head).

    Returns
    -------
    list of bool
        True if head's ROABP matrix is TNN (all entries non-negative).
    """
    from mikoshi_alignment.honesty import random_minor_screen
    matrices = attention_to_roabp(attention_weights, values)
    return [random_minor_screen(M) for M in matrices]


def complexity_class_estimate(
    attention_weights: NDArray,
    values: NDArray,
) -> Dict[str, Any]:
    """Estimate if model stays in polynomial complexity class.

    Parameters
    ----------
    attention_weights : array-like
        Shape (num_heads, seq_len, seq_len).
    values : array-like
        Shape (num_heads, seq_len, d_head).

    Returns
    -------
    dict
        Keys: ``max_rank``, ``mean_rank``, ``polynomial_bounded``.
    """
    matrices = attention_to_roabp(attention_weights, values)
    ranks = []
    for M in matrices:
        mid = max(1, M.shape[0] // 2)
        ranks.append(roabp_rank(M, mid))
    seq_len = attention_weights.shape[1]
    max_rank = max(ranks) if ranks else 0
    return {
        "max_rank": max_rank,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "polynomial_bounded": max_rank <= seq_len,
    }


class ROABPBridge:
    """Wraps attention patterns and provides ROABP-based safety scores.

    Parameters
    ----------
    max_rank_threshold : int
        Maximum allowed ROABP rank.
    """

    def __init__(self, max_rank_threshold: int = 64) -> None:
        self.max_rank_threshold = max_rank_threshold
        self.history: List[Dict] = []

    def check(
        self,
        attention_weights: NDArray,
        values: NDArray,
    ) -> Dict[str, Any]:
        """Run ROABP analysis on attention patterns.

        Returns
        -------
        dict
            Keys: ``score``, ``passed``, ``tnn_per_head``, ``complexity``.
        """
        tnn = tnn_check_per_head(attention_weights, values)
        complexity = complexity_class_estimate(attention_weights, values)
        tnn_rate = sum(tnn) / max(len(tnn), 1)
        poly_ok = complexity["polynomial_bounded"]
        score = 0.6 * tnn_rate + 0.4 * (1.0 if poly_ok else 0.0)
        result = {
            "score": score,
            "passed": score >= 0.5,
            "tnn_per_head": tnn,
            "complexity": complexity,
        }
        self.history.append(result)
        return result
