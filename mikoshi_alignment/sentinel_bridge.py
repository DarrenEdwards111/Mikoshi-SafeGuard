"""Improvement 6: Integration with Mikoshi Sentinel.

Two-Layer Safety: Sentinel verifies *actions*, Tri-Guard verifies *reasoning*.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray


class SentinelBridge:
    """Connects to Mikoshi Sentinel for action-level verification.

    Parameters
    ----------
    sentinel_url : str, optional
        HTTP endpoint for Sentinel.  If None, uses a local stub.
    """

    def __init__(self, sentinel_url: Optional[str] = None) -> None:
        self.sentinel_url = sentinel_url
        self._stub_mode = sentinel_url is None

    def verify_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an action through Sentinel.

        Parameters
        ----------
        action : dict
            Action description with keys like ``type``, ``target``, ``params``.

        Returns
        -------
        dict
            Keys: ``safe``, ``score``, ``reason``.
        """
        if self._stub_mode:
            # Stub: basic heuristic checks
            action_type = action.get("type", "unknown")
            dangerous = {"delete", "execute", "sudo", "rm", "drop", "shutdown"}
            if action_type.lower() in dangerous:
                return {"safe": False, "score": 0.0, "reason": f"Dangerous action type: {action_type}"}
            return {"safe": True, "score": 1.0, "reason": "Action passed stub checks"}

        try:
            import urllib.request
            import json
            data = json.dumps(action).encode()
            req = urllib.request.Request(
                f"{self.sentinel_url}/verify",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"safe": False, "score": 0.0, "reason": f"Sentinel unreachable: {e}"}

    def verify_reasoning(self, attributions: NDArray) -> Dict[str, Any]:
        """Verify reasoning through Tri-Guard (delegates to honesty check).

        Parameters
        ----------
        attributions : array-like
            Attribution matrix.

        Returns
        -------
        dict
            Keys: ``safe``, ``score``.
        """
        from mikoshi_alignment.honesty import honesty_score
        score = honesty_score(attributions)
        return {"safe": score >= 0.8, "score": score}

    def dual_verify(
        self,
        action: Dict[str, Any],
        attributions: NDArray,
    ) -> Dict[str, Any]:
        """Both Sentinel (actions) and Tri-Guard (reasoning).

        Returns
        -------
        dict
            Keys: ``safe``, ``action_result``, ``reasoning_result``, ``combined_score``.
        """
        action_result = self.verify_action(action)
        reasoning_result = self.verify_reasoning(attributions)
        combined = combined_safety_score(action_result, reasoning_result)
        return {
            "safe": action_result["safe"] and reasoning_result["safe"],
            "action_result": action_result,
            "reasoning_result": reasoning_result,
            "combined_score": combined,
        }


def combined_safety_score(
    sentinel_result: Dict[str, Any],
    tri_guard_result: Dict[str, Any],
) -> float:
    """Unified safety score from both layers.

    Parameters
    ----------
    sentinel_result : dict
        Must have ``score`` key.
    tri_guard_result : dict
        Must have ``score`` key.

    Returns
    -------
    float
        Combined score in [0, 1].  Uses geometric mean for strictness.
    """
    s1 = float(sentinel_result.get("score", 0.0))
    s2 = float(tri_guard_result.get("score", 0.0))
    return float(np.sqrt(max(0.0, s1) * max(0.0, s2)))


class TwoLayerSafety:
    """Wraps a model with both Sentinel (actions) and Tri-Guard (reasoning).

    Parameters
    ----------
    model : callable
        Model function: input → (action, attributions).
    sentinel_url : str, optional
        Sentinel endpoint.
    honesty_threshold : float
        Minimum honesty score.
    """

    def __init__(
        self,
        model: Callable,
        sentinel_url: Optional[str] = None,
        honesty_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.bridge = SentinelBridge(sentinel_url)
        self.honesty_threshold = honesty_threshold
        self.history: list = []

    def __call__(self, input_: Any) -> Dict[str, Any]:
        """Run model with two-layer safety verification.

        Parameters
        ----------
        input_ : Any
            Model input.

        Returns
        -------
        dict
            Keys: ``output``, ``safe``, ``verification``.
        """
        action, attributions = self.model(input_)
        verification = self.bridge.dual_verify(
            action if isinstance(action, dict) else {"type": "unknown", "value": action},
            attributions,
        )
        result = {
            "output": action,
            "safe": verification["safe"],
            "verification": verification,
        }
        self.history.append(result)
        return result
