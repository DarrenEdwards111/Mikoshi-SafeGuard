# Mikoshi AI Alignment

**Geometric safety verification for AI systems — grounded in N-Frame physics**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

---

## Overview

Mikoshi AI Alignment implements the **Tri-Guard** safety framework, a mathematically rigorous approach to AI alignment verification based on N-Frame theory (Edwards, 2023, 2024, 2025).

Unlike ad-hoc alignment approaches that rely on RLHF tuning or output filtering, Tri-Guard provides **geometric guarantees** at the reasoning level — verifying that a model's internal computations are honest, bounded, and exploit-free.

## The Three Guards

| Guard | What it checks | Mathematical basis |
|-------|---------------|-------------------|
| **Honesty** | Attribution matrices are totally non-negative (TNN) | Chapter 9: positivity ensures faithful reasoning |
| **Wall Stability** | Capability energy stays within budget | Israel thin-wall junction conditions |
| **Holonomy** | No cyclic reward hacking in update space | Flat connections / trivial holonomy |

## Six Improvements

Beyond the core three guards, this package includes:

1. **Deep Attribution** — Multi-method attribution (integrated gradients, attention decomposition, LRP) with cross-referencing to detect obfuscation
2. **Adversarial Stress Testing** — Systematic campaigns to find guard boundaries and gaps
3. **Temporal Drift Detection** — "Boiling frog" detection for slow cumulative drift and changepoint detection for sudden regime shifts
4. **Representation Monitoring** — Linear probes and sparse autoencoders to check if internal representations match external attributions
5. **ROABP Bridge** — Decompose transformer attention into Read-Once Algebraic Branching Programs for polynomial complexity bounds
6. **Sentinel Integration** — Two-Layer Safety combining Mikoshi Sentinel (action verification) with Tri-Guard (reasoning verification)

## Installation

```bash
pip install mikoshi-alignment
```

Or from source:

```bash
git clone https://github.com/DarrenEdwards111/Mikoshi-AI-Alignment.git
cd Mikoshi-AI-Alignment
pip install -e .
```

Optional dependencies:

```bash
pip install mikoshi-alignment[all]    # torch + scipy + matplotlib
pip install mikoshi-alignment[torch]  # deep attribution, representation monitoring
pip install mikoshi-alignment[viz]    # polytope visualization
```

## Quick Start

```python
import numpy as np
from mikoshi_alignment import TriGuard

# Create guard
guard = TriGuard(
    honesty_threshold=0.7,
    stability_budget=5.0,
    holonomy_tol=1e-4,
)

# Check model safety
result = guard.check(
    attribution_matrix=np.abs(np.random.randn(4, 4)),
    params=np.array([0.5, 0.3, 0.2]),
    update_history=np.random.randn(10, 3) * 0.01,
)

print(f"Safe: {result['safe']}, Score: {result['score']:.3f}")

# Enforce safety (raises RuntimeError if unsafe)
action = guard.enforce(
    attribution_matrix=np.eye(3),
    params=np.array([0.1, 0.1]),
    action={"type": "generate", "text": "Hello"},
)
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Tri-Guard                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Honesty │  │ Stability│  │ Holonomy │       │
│  │  Guard  │  │   Guard  │  │  Guard   │       │
│  │  (TNN)  │  │  (Wall)  │  │  (Flat)  │       │
│  └────┬────┘  └────┬─────┘  └────┬─────┘       │
│       └─────────┬──┴─────────────┘              │
│            Combined Score                        │
│       ┌─────────┴──────────┐                    │
│       │    Safe / Unsafe   │                    │
│       └────────────────────┘                    │
├─────────────────────────────────────────────────┤
│  Improvements:                                   │
│  • Deep Attribution    • Adversarial Testing     │
│  • Drift Detection     • Representation Monitor  │
│  • ROABP Bridge        • Sentinel Integration    │
└─────────────────────────────────────────────────┘
```

## Two-Layer Safety

Mikoshi AI Alignment is designed to work alongside [Mikoshi Sentinel](https://www.npmjs.com/package/mikoshi-sentinel) for **two-layer safety**:

| Layer | Tool | Verifies |
|-------|------|----------|
| **Actions** | Mikoshi Sentinel | What the model *does* (API calls, tool use, outputs) |
| **Reasoning** | Tri-Guard | How the model *thinks* (attributions, updates, representations) |

```python
from mikoshi_alignment.sentinel_bridge import TwoLayerSafety

safety = TwoLayerSafety(model, sentinel_url="http://localhost:3000")
result = safety(input_data)
# Both action-level AND reasoning-level verification
```

## Tri-Guard vs Ad-Hoc Alignment

| Approach | Guarantees | Level | Basis |
|----------|-----------|-------|-------|
| RLHF | Statistical | Behavioural | Human preferences |
| Output filtering | None | Surface | Pattern matching |
| Constitutional AI | Soft | Behavioural | Rules |
| **Tri-Guard** | **Geometric** | **Reasoning** | **N-Frame physics** |

Tri-Guard doesn't replace behavioural alignment — it adds a mathematically grounded verification layer beneath it.

## Theoretical Foundation

The framework is grounded in N-Frame theory:

- **Honesty** — Total non-negativity of Jacobian/attribution matrices (Chapter 9)
- **Stability** — Israel thin-wall junction conditions bounding capability energy
- **Holonomy** — Flat connections ensuring path-independence in update space
- **SPDP Polytope** — Shifted Partial Derivative matrices defining admissible inference regions (Chapter 7)

See [paper/README.md](paper/README.md) for references.

### References

- Edwards, D. (2023). *N-Frame: A Geometric Framework for Physics and Information*
- Edwards, D. (2024). *Extensions to N-Frame Theory: Applications in AI Safety*
- Edwards, D. (2025). *Tri-Guard: Geometric Safety Verification for AI Systems*

## API Reference

### Core Guards

- `HonestyGuard(threshold=0.8)` — TNN-based attribution verification
- `WallStabilityGuard(budget=1.0)` — Capability energy bounding
- `HolonomyGuard(tol=1e-6)` — Reward-hacking detection

### Combined

- `TriGuard(honesty_threshold, stability_budget, holonomy_tol)` — All three guards
  - `.check(attribution_matrix, params, update_history)` — Run all checks
  - `.enforce(attribution_matrix, params, action)` — Block unsafe actions
  - `.wrap_model(model)` — Safety-wrapped model
  - `.score()` / `.is_safe()` / `.report()` — Results

### Improvements

- `mikoshi_alignment.deep_attribution` — Multi-method attribution
- `mikoshi_alignment.adversarial` — Stress testing
- `mikoshi_alignment.drift` — Temporal drift detection
- `mikoshi_alignment.representation` — Internal monitoring
- `mikoshi_alignment.roabp_bridge` — ROABP analysis
- `mikoshi_alignment.sentinel_bridge` — Sentinel integration
- `mikoshi_alignment.polytope` — SPDP polytope geometry

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Credits

Developed by **Mikoshi Ltd**.

Theoretical foundation: N-Frame theory (Edwards, 2023, 2024, 2025).
