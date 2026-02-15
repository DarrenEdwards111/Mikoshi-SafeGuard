"""Example: wrapping an LLM with Tri-Guard safety."""

import numpy as np
from mikoshi_alignment import TriGuard

# Simulated LLM
def fake_llm(prompt: str) -> str:
    return f"Response to: {prompt}"

# Create guard
guard = TriGuard(honesty_threshold=0.6, stability_budget=10.0)

# Wrap the model
def safe_llm(prompt: str) -> str:
    # In production, extract real attributions and params
    J = np.abs(np.random.randn(5, 5))
    params = np.random.randn(10) * 0.1
    result = guard.check(J, params)
    if not result["safe"]:
        return "[BLOCKED] Response failed safety verification."
    return fake_llm(prompt)

# Test
print(safe_llm("Tell me about AI safety"))
print(f"\nGuard score: {guard.score():.3f}")
print(f"Full report: {guard.report()}")
