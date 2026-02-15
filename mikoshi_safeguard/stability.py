"""Guard 2: Wall Stability — capability bounding via Israel junction conditions.

The Israel thin-wall formalism bounds the energy on a domain wall separating
two space-time regions.  Here, the *interior* is the model's capability space
and the *exterior* is the safety boundary.  The wall's tension must balance
curvature to prevent capability escape.

Reference: geometric safety theory.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def capability_energy(params: NDArray, metric: Literal["l2", "linf", "l1"] = "l2") -> float:
    """Compute capability energy from parameter vector.

    Parameters
    ----------
    params : array-like
        Model parameters (flattened).
    metric : str
        Norm to use: ``'l2'``, ``'l1'``, or ``'linf'``.

    Returns
    -------
    float
        Non-negative energy scalar.
    """
    p = np.asarray(params, dtype=float).ravel()
    if metric == "l2":
        return float(np.linalg.norm(p, 2))
    elif metric == "l1":
        return float(np.linalg.norm(p, 1))
    elif metric == "linf":
        return float(np.linalg.norm(p, np.inf))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def safety_tension(budget: float, current_energy: float) -> float:
    """Compute safety tension — how much margin remains before breach.

    Parameters
    ----------
    budget : float
        Maximum allowed capability energy.
    current_energy : float
        Current capability energy.

    Returns
    -------
    float
        Positive = safe, zero/negative = breach.
    """
    return budget - current_energy


def barrier_lyapunov(energy: float, budget: float, margin: float = 0.1) -> float:
    """Lyapunov barrier function.  Diverges as energy → budget.

    Parameters
    ----------
    energy : float
        Current capability energy.
    budget : float
        Maximum allowed energy.
    margin : float
        Soft margin before the hard wall.

    Returns
    -------
    float
        Barrier value.  +inf if energy ≥ budget.
    """
    gap = budget - energy
    if gap <= 0:
        return float("inf")
    return -np.log(gap / (budget + margin))


def curvature_check(trajectory: NDArray, budget: float) -> bool:
    """Check if the capability trajectory's curvature exceeds the barrier.

    Uses second-order finite differences on the energy trajectory.

    Parameters
    ----------
    trajectory : array-like
        Sequence of parameter snapshots, shape (T, D).
    budget : float
        Energy budget.

    Returns
    -------
    bool
        True if curvature is within safe bounds.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    energies = np.array([capability_energy(p) for p in trajectory])
    if len(energies) < 3:
        return True
    # Second derivative (curvature proxy)
    curvature = np.diff(energies, n=2)
    # If curvature is consistently positive and energy nears budget → unsafe
    for i, c in enumerate(curvature):
        if c > 0 and energies[i + 2] > 0.9 * budget:
            return False
    return True


def israel_junction_check(
    interior_curvature: float,
    exterior_curvature: float,
    tension: float,
) -> bool:
    """Israel thin-wall junction condition.

    The jump in extrinsic curvature across the wall must equal 8πG × tension.
    Simplified: |K_ext - K_int| ≤ tension.

    Parameters
    ----------
    interior_curvature : float
        Curvature on the capability side.
    exterior_curvature : float
        Curvature on the safety side.
    tension : float
        Wall tension (safety budget margin).

    Returns
    -------
    bool
        True if the junction condition is satisfied (wall is stable).
    """
    jump = abs(exterior_curvature - interior_curvature)
    return jump <= tension


def stability_score(
    params: NDArray,
    budget: float,
    trajectory: Optional[NDArray] = None,
) -> float:
    """Combined stability score in [0, 1].

    Parameters
    ----------
    params : array-like
        Current parameters.
    budget : float
        Energy budget.
    trajectory : array-like, optional
        Historical parameter snapshots.

    Returns
    -------
    float
        1.0 = fully stable, 0.0 = breached.
    """
    energy = capability_energy(params)
    tension = safety_tension(budget, energy)
    if tension <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, tension / budget))
    if trajectory is not None:
        curv_ok = curvature_check(trajectory, budget)
        if not curv_ok:
            ratio *= 0.5
    return float(ratio)


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------

class WallStabilityGuard:
    """Monitors parameter trajectories and alerts on capability escape.

    Parameters
    ----------
    budget : float
        Maximum allowed capability energy.
    metric : str
        Norm for energy computation.
    margin : float
        Soft margin for barrier function.
    """

    def __init__(
        self,
        budget: float = 1.0,
        metric: Literal["l2", "linf", "l1"] = "l2",
        margin: float = 0.1,
    ) -> None:
        self.budget = budget
        self.metric = metric
        self.margin = margin
        self.trajectory: List[NDArray] = []
        self.history: List[Dict] = []

    def check(self, params: NDArray) -> Dict:
        """Check stability of current parameters.

        Returns
        -------
        dict
            Keys: ``score``, ``passed``, ``energy``, ``tension``, ``barrier``.
        """
        params = np.asarray(params, dtype=float)
        self.trajectory.append(params.copy())
        energy = capability_energy(params, self.metric)
        tension = safety_tension(self.budget, energy)
        barrier = barrier_lyapunov(energy, self.budget, self.margin)
        traj = np.array(self.trajectory) if len(self.trajectory) >= 3 else None
        score = stability_score(params, self.budget, traj)
        result = {
            "score": score,
            "passed": score > 0.0,
            "energy": energy,
            "tension": tension,
            "barrier": barrier,
        }
        self.history.append(result)
        return result

    def score(self, params: NDArray) -> float:
        """Return stability score for *params*."""
        traj = np.array(self.trajectory) if len(self.trajectory) >= 3 else None
        return stability_score(params, self.budget, traj)

    def is_safe(self, params: NDArray) -> bool:
        """Return True if *params* are within the stability wall."""
        return self.check(params)["passed"]
