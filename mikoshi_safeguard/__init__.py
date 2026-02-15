"""
Mikoshi SafeGuard — Runtime safety verification for AI systems (geometric).

Tri-Guard geometric safety framework.

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

from mikoshi_safeguard.tri_guard import TriGuard
from mikoshi_safeguard.honesty import HonestyGuard
from mikoshi_safeguard.stability import WallStabilityGuard
from mikoshi_safeguard.holonomy import HolonomyGuard

__all__ = [
    "TriGuard",
    "HonestyGuard",
    "WallStabilityGuard",
    "HolonomyGuard",
    "__version__",
]
