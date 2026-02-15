"""Improvement 3: Temporal Drift Detection.

Detect slow cumulative drift ("boiling frog") and sudden regime changes
in model parameters and attributions over time.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray


def exponential_moving_drift(history: NDArray, alpha: float = 0.01) -> NDArray:
    """Exponential moving average of parameter-change magnitudes.

    Parameters
    ----------
    history : array-like
        Shape (T, D) — parameter snapshots over time.
    alpha : float
        Smoothing factor.

    Returns
    -------
    NDArray
        EMA of per-step drift magnitudes, shape (T-1,).
    """
    H = np.asarray(history, dtype=float)
    if H.shape[0] < 2:
        return np.array([0.0])
    diffs = np.linalg.norm(np.diff(H, axis=0), axis=-1)
    ema = np.zeros_like(diffs)
    ema[0] = diffs[0]
    for i in range(1, len(diffs)):
        ema[i] = alpha * diffs[i] + (1 - alpha) * ema[i - 1]
    return ema


def changepoint_detection(
    signal: NDArray,
    method: Literal["cusum", "threshold"] = "cusum",
    threshold: float = 1.0,
) -> List[int]:
    """Detect sudden regime changes in a 1-D signal.

    Parameters
    ----------
    signal : array-like
        1-D signal.
    method : str
        ``'cusum'`` — cumulative sum control chart.
        ``'threshold'`` — simple threshold on differences.
    threshold : float
        Detection threshold.

    Returns
    -------
    list of int
        Indices of detected changepoints.
    """
    s = np.asarray(signal, dtype=float).ravel()
    changepoints: List[int] = []
    if method == "cusum":
        mean = s.mean()
        cusum_pos = 0.0
        cusum_neg = 0.0
        for i in range(len(s)):
            cusum_pos = max(0.0, cusum_pos + s[i] - mean)
            cusum_neg = max(0.0, cusum_neg - s[i] + mean)
            if cusum_pos > threshold or cusum_neg > threshold:
                changepoints.append(i)
                cusum_pos = 0.0
                cusum_neg = 0.0
    elif method == "threshold":
        diffs = np.abs(np.diff(s))
        for i, d in enumerate(diffs):
            if d > threshold:
                changepoints.append(i + 1)
    return changepoints


class DriftDetector:
    """Tracks model behaviour over time and detects drift.

    Parameters
    ----------
    threshold : float
        Boiling-frog threshold for cumulative drift.
    window : int
        Window size for drift computation.
    """

    def __init__(self, threshold: float = 0.1, window: int = 100) -> None:
        self.threshold = threshold
        self.window = window
        self.params_history: List[NDArray] = []
        self.attribution_history: List[NDArray] = []
        self.output_history: List[Any] = []
        self.steps: List[int] = []

    def record(
        self,
        step: int,
        params: NDArray,
        attributions: Optional[NDArray] = None,
        outputs: Optional[Any] = None,
    ) -> None:
        """Log a timestep."""
        self.steps.append(step)
        self.params_history.append(np.asarray(params, dtype=float).ravel())
        if attributions is not None:
            self.attribution_history.append(np.asarray(attributions, dtype=float).ravel())
        if outputs is not None:
            self.output_history.append(outputs)

    def cumulative_drift(self, window: Optional[int] = None) -> float:
        """Total parameter drift over *window* steps.

        Returns
        -------
        float
            Cumulative L2 drift.
        """
        w = window or self.window
        if len(self.params_history) < 2:
            return 0.0
        recent = self.params_history[-w:]
        diffs = np.linalg.norm(np.diff(recent, axis=0), axis=1)
        return float(np.sum(diffs))

    def detect_boiling_frog(self, threshold: Optional[float] = None) -> bool:
        """Detect if each step is small but cumulative drift is large.

        Returns
        -------
        bool
            True if boiling-frog drift detected.
        """
        t = threshold or self.threshold
        if len(self.params_history) < 3:
            return False
        recent = self.params_history[-self.window :]
        diffs = np.linalg.norm(np.diff(recent, axis=0), axis=1)
        max_step = float(np.max(diffs))
        total = float(np.sum(diffs))
        # Each step is small but total is large
        return max_step < t and total > t * 10

    def safe_region_shift(
        self,
        initial_region: NDArray,
        current_region: NDArray,
    ) -> float:
        """Measure how much the safe region has shifted.

        Parameters
        ----------
        initial_region : array-like
            Centre of initial safe region.
        current_region : array-like
            Centre of current safe region.

        Returns
        -------
        float
            L2 distance between region centres.
        """
        return float(np.linalg.norm(
            np.asarray(initial_region, dtype=float) - np.asarray(current_region, dtype=float)
        ))

    def drift_score(self) -> float:
        """Combined drift score in [0, 1].  1 = no drift.

        Returns
        -------
        float
            Drift score.
        """
        if len(self.params_history) < 2:
            return 1.0
        total = self.cumulative_drift()
        # Sigmoid decay
        return float(1.0 / (1.0 + total))
