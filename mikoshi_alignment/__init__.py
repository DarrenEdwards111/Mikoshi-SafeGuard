"""
Mikoshi AI Alignment — Geometric safety verification for AI systems.

Tri-Guard framework grounded in N-Frame physics (Edwards, 2023, 2024, 2025).

Three guards:
    1. Honesty Guard — positivity / TNN checks on attribution matrices
    2. Wall Stability Guard — capability bounding via Israel junction conditions
    3. Holonomy Guard — reward-hacking detection via flat-connection checks

Six improvements:
    4. Deep attribution methods
    5. Adversarial stress testing
    6. Temporal drift detection
    7. Representation-level monitoring
    8. ROABP approximation for transformers
    9. Integration with Mikoshi Sentinel

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

__version__ = "0.1.0"

from mikoshi_alignment.tri_guard import TriGuard
from mikoshi_alignment.honesty import HonestyGuard
from mikoshi_alignment.stability import WallStabilityGuard
from mikoshi_alignment.holonomy import HolonomyGuard

__all__ = [
    "TriGuard",
    "HonestyGuard",
    "WallStabilityGuard",
    "HolonomyGuard",
    "__version__",
]
