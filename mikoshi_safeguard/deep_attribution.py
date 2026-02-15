"""Improvement 1: Deep Attribution Methods.

Multi-method attribution for cross-referencing model explanations.
Requires optional ``torch`` dependency for model-based methods;
standalone functions work with numpy arrays.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


def integrated_gradients(
    model: Callable[[NDArray], NDArray],
    input_: NDArray,
    baseline: Optional[NDArray] = None,
    steps: int = 50,
) -> NDArray:
    """Path-integrated gradients (Sundararajan et al., 2017).

    Parameters
    ----------
    model : callable
        Function from input array to scalar output.
    input_ : array-like
        Input point.
    baseline : array-like, optional
        Baseline (default: zeros).
    steps : int
        Number of interpolation steps.

    Returns
    -------
    NDArray
        Attribution vector, same shape as *input_*.
    """
    x = np.asarray(input_, dtype=float)
    b = np.zeros_like(x) if baseline is None else np.asarray(baseline, dtype=float)
    eps = 1e-5
    grads = np.zeros_like(x)
    for k in range(steps):
        alpha = k / steps
        interp = b + alpha * (x - b)
        # Numerical gradient at interpolation point
        grad_k = np.zeros_like(x)
        for i in range(x.size):
            x_plus = interp.copy()
            x_minus = interp.copy()
            x_plus.flat[i] += eps
            x_minus.flat[i] -= eps
            grad_k.flat[i] = (float(np.sum(model(x_plus))) - float(np.sum(model(x_minus)))) / (2 * eps)
        grads += grad_k
    return (x - b) * grads / steps


def attention_head_decomposition(
    attention_weights: NDArray,
    values: NDArray,
) -> NDArray:
    """Per-head attribution from attention weights and values.

    Parameters
    ----------
    attention_weights : array-like
        Shape (num_heads, seq_len, seq_len).
    values : array-like
        Shape (num_heads, seq_len, d_head).

    Returns
    -------
    NDArray
        Per-head attribution scores, shape (num_heads,).
    """
    W = np.asarray(attention_weights, dtype=float)
    V = np.asarray(values, dtype=float)
    num_heads = W.shape[0]
    scores = np.zeros(num_heads)
    for h in range(num_heads):
        # Attribution = Frobenius norm of attention-weighted values
        context = W[h] @ V[h]
        scores[h] = float(np.linalg.norm(context, "fro"))
    return scores


def multi_method_attribution(
    attributions_dict: Dict[str, NDArray],
) -> Dict[str, Any]:
    """Cross-reference multiple attribution methods.

    Parameters
    ----------
    attributions_dict : dict
        Mapping method_name → attribution vector.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``agreement_score``, ``methods``.
    """
    methods = list(attributions_dict.keys())
    arrays = [np.asarray(v, dtype=float).ravel() for v in attributions_dict.values()]
    if not arrays:
        return {"mean": np.array([]), "std": np.array([]), "agreement_score": 1.0, "methods": []}
    stacked = np.array(arrays)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    agreement = attribution_agreement_score(attributions_dict)
    return {"mean": mean, "std": std, "agreement_score": agreement, "methods": methods}


def attribution_agreement_score(attributions_dict: Dict[str, NDArray]) -> float:
    """Measure agreement across attribution methods.

    High disagreement is a red flag — the model may be obfuscating.

    Parameters
    ----------
    attributions_dict : dict
        Mapping method_name → attribution vector.

    Returns
    -------
    float
        Score in [0, 1].  1.0 = perfect agreement.
    """
    arrays = [np.asarray(v, dtype=float).ravel() for v in attributions_dict.values()]
    if len(arrays) < 2:
        return 1.0
    # Pairwise cosine similarities
    from mikoshi_safeguard.utils import cosine_similarity
    sims = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            sims.append(cosine_similarity(arrays[i], arrays[j]))
    return float(np.mean(sims))


def layer_wise_relevance(
    activations: List[NDArray],
    weights: List[NDArray],
    target_index: int = 0,
) -> List[NDArray]:
    """Simplified Layer-wise Relevance Propagation (LRP).

    Parameters
    ----------
    activations : list of NDArray
        Layer activations, from input to output.
    weights : list of NDArray
        Weight matrices between layers.
    target_index : int
        Output neuron index to attribute.

    Returns
    -------
    list of NDArray
        Relevance scores per layer, from output back to input.
    """
    n_layers = len(activations)
    relevances: List[NDArray] = [np.zeros(1)] * n_layers
    # Start from last layer
    last = np.asarray(activations[-1], dtype=float).ravel()
    R = np.zeros_like(last)
    if target_index < len(R):
        R[target_index] = last[target_index]
    relevances[-1] = R

    for l in range(n_layers - 2, -1, -1):
        a = np.asarray(activations[l], dtype=float).ravel()
        w = np.asarray(weights[l], dtype=float)
        # z_j = sum_i a_i * w_ij
        z = a[:, None] * w + 1e-9
        s = z.sum(axis=0)
        s[s == 0] = 1e-9
        R_new = np.sum((z / s[None, :]) * R[None, :], axis=1)
        relevances[l] = R_new
        R = R_new

    return relevances
