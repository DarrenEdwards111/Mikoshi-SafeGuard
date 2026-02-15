"""Example: monitoring model training with drift detection."""

import numpy as np
from mikoshi_safeguard.drift import DriftDetector
from mikoshi_safeguard.stability import WallStabilityGuard

# Setup
drift = DriftDetector(threshold=0.05)
wall = WallStabilityGuard(budget=5.0)

# Simulate training
print("Training with drift monitoring...\n")
for step in range(50):
    params = np.random.randn(10) * (0.1 + step * 0.005)
    drift.record(step, params)
    stability = wall.check(params)

    if step % 10 == 0:
        print(f"Step {step:3d} | Drift: {drift.drift_score():.3f} | "
              f"Stability: {stability['score']:.3f} | "
              f"Energy: {stability['energy']:.3f}")

print(f"\nFinal drift score: {drift.drift_score():.3f}")
print(f"Boiling frog detected: {drift.detect_boiling_frog()}")
