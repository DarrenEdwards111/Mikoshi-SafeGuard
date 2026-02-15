"""Improvement 2: Adversarial Stress Testing.

Systematic adversarial campaigns against all three guards.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_adversarial_inputs(
    guard_check: Callable[[NDArray], bool],
    shape: Tuple[int, ...],
    n: int = 100,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """Generate inputs designed to probe guard boundaries.

    Parameters
    ----------
    guard_check : callable
        Function that takes an array and returns True (safe) or False.
    shape : tuple
        Shape of each input.
    n : int
        Number of adversarial inputs to generate.
    seed : int, optional
        Random seed.

    Returns
    -------
    list of NDArray
        Inputs that are near the guard boundary (barely pass or barely fail).
    """
    rng = np.random.default_rng(seed)
    boundary_inputs: List[NDArray] = []
    for _ in range(n):
        # Generate random input
        x = rng.standard_normal(shape)
        # Try to find boundary by scaling
        safe = guard_check(x)
        scale_lo, scale_hi = 0.0, 2.0
        for __ in range(20):
            mid = (scale_lo + scale_hi) / 2.0
            x_scaled = x * mid
            if guard_check(x_scaled) == safe:
                scale_lo = mid
            else:
                scale_hi = mid
        boundary_inputs.append(x * (scale_lo + scale_hi) / 2.0)
    return boundary_inputs


def fuzz_attribution(
    attribution: NDArray,
    epsilon: float = 0.01,
    n: int = 50,
    seed: Optional[int] = None,
) -> List[Tuple[NDArray, bool]]:
    """Small perturbations to check if attribution signs flip.

    Parameters
    ----------
    attribution : array-like
        Attribution matrix.
    epsilon : float
        Perturbation magnitude.
    n : int
        Number of perturbations.
    seed : int, optional
        Random seed.

    Returns
    -------
    list of (NDArray, bool)
        Perturbed attributions and whether sign flipped (True = flipped).
    """
    A = np.asarray(attribution, dtype=float)
    rng = np.random.default_rng(seed)
    original_signs = np.sign(A)
    results: List[Tuple[NDArray, bool]] = []
    for _ in range(n):
        noise = rng.standard_normal(A.shape) * epsilon
        perturbed = A + noise
        flipped = bool(np.any(np.sign(perturbed) != original_signs))
        results.append((perturbed, flipped))
    return results


def guard_robustness_score(
    guard_check: Callable[[NDArray], bool],
    test_suite: List[NDArray],
) -> float:
    """Measure guard robustness over a test suite.

    Parameters
    ----------
    guard_check : callable
        Guard check function.
    test_suite : list of NDArray
        Test inputs.

    Returns
    -------
    float
        Fraction of test inputs that yield consistent results under small perturbation.
    """
    if not test_suite:
        return 1.0
    consistent = 0
    for x in test_suite:
        x = np.asarray(x, dtype=float)
        base = guard_check(x)
        # Check 5 small perturbations
        same = sum(
            1 for _ in range(5)
            if guard_check(x + np.random.randn(*x.shape) * 1e-4) == base
        )
        if same >= 4:
            consistent += 1
    return consistent / len(test_suite)


class StressTest:
    """Systematic adversarial campaign against guards.

    Parameters
    ----------
    guards : dict
        Mapping guard_name → check callable.
    """

    def __init__(self, guards: Dict[str, Callable[[NDArray], bool]]) -> None:
        self.guards = guards
        self.results: Dict[str, List[Dict]] = {name: [] for name in guards}

    def run(
        self,
        shape: Tuple[int, ...],
        n_inputs: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run stress tests on all guards.

        Returns
        -------
        dict
            Guard name → robustness score.
        """
        rng = np.random.default_rng(seed)
        scores: Dict[str, float] = {}
        for name, check in self.guards.items():
            test_inputs = [rng.standard_normal(shape) for _ in range(n_inputs)]
            score = guard_robustness_score(check, test_inputs)
            scores[name] = score
            self.results[name].append({"n_inputs": n_inputs, "robustness": score})
        return scores

    def report(self) -> Dict[str, Any]:
        """Return full results."""
        return dict(self.results)


def find_guard_gaps(
    guard_checks: Dict[str, Callable[[NDArray], bool]],
    shape: Tuple[int, ...],
    n: int = 200,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """Find inputs that pass ALL guards — potential gaps.

    Parameters
    ----------
    guard_checks : dict
        Mapping name → check function.
    shape : tuple
        Input shape.
    n : int
        Number of random probes.
    seed : int, optional
        Random seed.

    Returns
    -------
    list of NDArray
        Inputs that pass every guard.
    """
    rng = np.random.default_rng(seed)
    gaps: List[NDArray] = []
    for _ in range(n):
        x = rng.standard_normal(shape) * 2
        if all(check(x) for check in guard_checks.values()):
            gaps.append(x)
    return gaps
