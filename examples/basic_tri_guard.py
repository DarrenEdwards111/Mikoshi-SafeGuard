"""Basic Tri-Guard usage example."""

import numpy as np
from mikoshi_alignment import TriGuard

# Create a Tri-Guard instance
guard = TriGuard(
    honesty_threshold=0.7,
    stability_budget=5.0,
    holonomy_tol=1e-4,
)

# Simulate model outputs
attribution_matrix = np.abs(np.random.randn(4, 4))  # Positive = honest
params = np.array([0.5, 0.3, 0.2, 0.1])             # Within budget
updates = np.array([                                   # Linear updates = flat
    [0.1, 0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2, 0.2],
    [0.3, 0.3, 0.3, 0.3],
    [0.4, 0.4, 0.4, 0.4],
])

# Run the check
result = guard.check(attribution_matrix, params, updates)
print(f"Safe: {result['safe']}")
print(f"Score: {result['score']:.3f}")
print(f"Honesty: {result['honesty']['score']:.3f}")
print(f"Stability: {result['stability']['score']:.3f}")
print(f"Holonomy: {result['holonomy']['score']:.3f}")

# Enforce safety
try:
    action = guard.enforce(attribution_matrix, params, {"type": "generate", "text": "Hello"}, updates)
    print(f"\nAction allowed: {action}")
except RuntimeError as e:
    print(f"\nAction blocked: {e}")
