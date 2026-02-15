"""Example: Two-Layer Safety with Sentinel + Tri-Guard."""

import numpy as np
from mikoshi_safeguard.sentinel_bridge import SentinelBridge, TwoLayerSafety

# Create bridge (stub mode — no Sentinel server needed)
bridge = SentinelBridge()

# Test action verification
safe_action = {"type": "generate", "target": "text"}
dangerous_action = {"type": "delete", "target": "database"}

print("Action verification:")
print(f"  Generate: {bridge.verify_action(safe_action)}")
print(f"  Delete:   {bridge.verify_action(dangerous_action)}")

# Test dual verification
attributions = np.abs(np.random.randn(4, 4))
result = bridge.dual_verify(safe_action, attributions)
print(f"\nDual verification: safe={result['safe']}, score={result['combined_score']:.3f}")

# Two-Layer Safety wrapper
def fake_model(input_):
    action = {"type": "generate", "value": f"output for {input_}"}
    attributions = np.abs(np.random.randn(3, 3))
    return action, attributions

tls = TwoLayerSafety(fake_model)
result = tls({"prompt": "Hello"})
print(f"\nTwo-Layer result: safe={result['safe']}, output={result['output']}")
