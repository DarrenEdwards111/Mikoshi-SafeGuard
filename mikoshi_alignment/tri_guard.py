"""Combined Tri-Guard Runtime Filter.

The main entry point for Mikoshi AI Alignment safety verification.
Combines all three guards (Honesty, Stability, Holonomy) into a single
runtime filter with configurable thresholds and full audit trail.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from mikoshi_alignment.honesty import HonestyGuard
from mikoshi_alignment.stability import WallStabilityGuard
from mikoshi_alignment.holonomy import HolonomyGuard


class TriGuard:
    """Combined Tri-Guard runtime safety filter.

    Parameters
    ----------
    honesty_threshold : float
        Minimum honesty score to pass.
    stability_budget : float
        Maximum capability energy.
    holonomy_tol : float
        Flatness tolerance for holonomy checks.
    """

    def __init__(
        self,
        honesty_threshold: float = 0.8,
        stability_budget: float = 1.0,
        holonomy_tol: float = 1e-6,
    ) -> None:
        self.honesty_guard = HonestyGuard(threshold=honesty_threshold)
        self.stability_guard = WallStabilityGuard(budget=stability_budget)
        self.holonomy_guard = HolonomyGuard(tol=holonomy_tol)
        self.history: List[Dict[str, Any]] = []
        self._last_result: Optional[Dict[str, Any]] = None

    def check(
        self,
        attribution_matrix: NDArray,
        params: NDArray,
        update_history: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        """Run all three guards.

        Parameters
        ----------
        attribution_matrix : array-like
            Attribution / Jacobian matrix for honesty check.
        params : array-like
            Current model parameters for stability check.
        update_history : array-like, optional
            History of parameter updates for holonomy check.

        Returns
        -------
        dict
            Keys: ``safe``, ``score``, ``honesty``, ``stability``, ``holonomy``.
        """
        honesty_result = self.honesty_guard.check(attribution_matrix)
        stability_result = self.stability_guard.check(params)
        holonomy_result = self.holonomy_guard.check(update_history)

        combined_score = (
            honesty_result["score"] * 0.4
            + stability_result["score"] * 0.3
            + holonomy_result["score"] * 0.3
        )
        safe = (
            honesty_result["passed"]
            and stability_result["passed"]
            and holonomy_result["passed"]
        )

        result = {
            "safe": safe,
            "score": combined_score,
            "honesty": honesty_result,
            "stability": stability_result,
            "holonomy": holonomy_result,
        }
        self._last_result = result
        self.history.append(result)
        return result

    def score(self) -> float:
        """Return the most recent combined safety score.

        Returns
        -------
        float
            Score in [0, 1], or 0.0 if no checks have been run.
        """
        if self._last_result is None:
            return 0.0
        return float(self._last_result["score"])

    def is_safe(self) -> bool:
        """Return True if the last check passed all guards.

        Returns
        -------
        bool
        """
        if self._last_result is None:
            return False
        return bool(self._last_result["safe"])

    def report(self) -> Dict[str, Any]:
        """Return detailed report from the last check.

        Returns
        -------
        dict
            Full per-guard results, or empty dict if no checks run.
        """
        if self._last_result is None:
            return {}
        return dict(self._last_result)

    def enforce(
        self,
        attribution_matrix: NDArray,
        params: NDArray,
        action: Any,
        update_history: Optional[NDArray] = None,
    ) -> Any:
        """Block unsafe actions.  Returns action if safe, raises otherwise.

        Parameters
        ----------
        attribution_matrix : array-like
            Attribution matrix.
        params : array-like
            Model parameters.
        action : Any
            The action to potentially block.
        update_history : array-like, optional
            Update history.

        Returns
        -------
        Any
            The action, if it passes all guards.

        Raises
        ------
        RuntimeError
            If any guard fails.
        """
        result = self.check(attribution_matrix, params, update_history)
        if not result["safe"]:
            failed = []
            if not result["honesty"]["passed"]:
                failed.append("honesty")
            if not result["stability"]["passed"]:
                failed.append("stability")
            if not result["holonomy"]["passed"]:
                failed.append("holonomy")
            raise RuntimeError(
                f"Tri-Guard blocked action: failed guards = {failed}, "
                f"score = {result['score']:.3f}"
            )
        return action

    def wrap_model(
        self,
        model: Callable,
        param_extractor: Optional[Callable] = None,
        attribution_extractor: Optional[Callable] = None,
    ) -> Callable:
        """Return a safety-wrapped version of the model.

        Parameters
        ----------
        model : callable
            The model to wrap.
        param_extractor : callable, optional
            Function to extract parameters from the model.
        attribution_extractor : callable, optional
            Function to extract attributions given (model, input).

        Returns
        -------
        callable
            Wrapped model that runs Tri-Guard before returning output.
        """
        guard = self

        def wrapped(input_: Any) -> Any:
            output = model(input_)
            # Extract safety-relevant information
            if attribution_extractor:
                J = attribution_extractor(model, input_)
            else:
                J = np.eye(3)  # Placeholder
            if param_extractor:
                params = param_extractor(model)
            else:
                params = np.zeros(3)  # Placeholder
            result = guard.check(J, params)
            if not result["safe"]:
                raise RuntimeError(f"Tri-Guard blocked output: score = {result['score']:.3f}")
            return output

        return wrapped
