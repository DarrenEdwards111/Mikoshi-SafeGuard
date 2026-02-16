"""Improvement 6: Integration with Mikoshi Sentinel.

Two-Layer Safety: Sentinel verifies *actions*, Tri-Guard verifies *reasoning*.
Now uses the native Python Sentinel implementation (no JS dependency needed).

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from mikoshi_safeguard.sentinel.engine import Sentinel


class SentinelBridge:
    """Connects to Mikoshi Sentinel for action-level verification.

    Now uses the native Python Sentinel engine directly.

    Parameters
    ----------
    sentinel : Sentinel, optional
        Pre-configured Sentinel instance. If None, creates one with defaults.
    sentinel_url : str, optional
        Deprecated. Kept for backwards compatibility but ignored when
        the native sentinel is available.
    """

    def __init__(self, sentinel: Optional[Sentinel] = None,
                 sentinel_url: Optional[str] = None) -> None:
        self.sentinel = sentinel or Sentinel(enable_intent_verification=False)
        self._sentinel_url = sentinel_url  # kept for compat

    def verify_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an action through Sentinel.

        Parameters
        ----------
        action : dict
            Action description with keys like ``tool``/``type``, ``args``/``params``.

        Returns
        -------
        dict
            Keys: ``safe``, ``score``, ``reason``.
        """
        # Build a raw action in Sentinel's expected format
        raw_action = {
            'tool': action.get('tool') or action.get('type', 'unknown'),
            'args': action.get('args') or action.get('params', {}),
        }

        # Run async verify in sync context
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                verdict = pool.submit(
                    lambda: asyncio.run(self.sentinel.verify(raw_action))
                ).result()
        else:
            verdict = asyncio.run(self.sentinel.verify(raw_action))

        return {
            'safe': verdict['allowed'],
            'score': verdict['confidence'],
            'reason': verdict['violations'][0]['reason'] if verdict['violations'] else 'Action passed all checks',
        }

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
        from mikoshi_safeguard.honesty import honesty_score
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
    sentinel : Sentinel, optional
        Pre-configured Sentinel instance.
    honesty_threshold : float
        Minimum honesty score.
    """

    def __init__(
        self,
        model: Callable,
        sentinel: Optional[Sentinel] = None,
        honesty_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.bridge = SentinelBridge(sentinel=sentinel)
        self.honesty_threshold = honesty_threshold
        self.history: list = []

    def __call__(self, input_: Any) -> Dict[str, Any]:
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
