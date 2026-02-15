# Mikoshi SafeGuard

**Runtime safety verification for AI systems (geometric)**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

---

## Overview

Mikoshi SafeGuard implements the **Tri-Guard** safety framework — runtime verification that checks whether an AI system's reasoning is honest, bounded, and exploit-free.

Unlike ad-hoc alignment approaches that rely on RLHF tuning or output filtering, Tri-Guard verifies safety **at the reasoning level** using mathematical geometry.

## Why Geometric?

Most AI safety tools check *what a model says*. SafeGuard checks *how it thinks* — using geometry to verify the mathematical structure of its reasoning.

**Honesty → Positivity in matrix space.** When an AI explains its decisions, those explanations form an attribution matrix. If the matrix is "totally non-negative" (all minors ≥ 0), the explanation is faithful — no hidden sign cancellations, no deceptive reasoning. This is a geometric property: the matrix must lie inside a specific region (the positive cone) of matrix space. Step outside, and the model is hiding something.

**Stability → Curved boundaries in parameter space.** An AI's capabilities can be measured as energy in a parameter space. Safety means that energy stays inside a budget — bounded by a Lyapunov barrier surface. Think of it as a curved wall: if the model's capability trajectory hits the wall, it's escaping its safety bounds. The wall's curvature comes from the same physics that governs bubble stability in cosmology (Israel junction conditions).

**Consistency → Curvature on a manifold.** When an AI updates its behaviour over a sequence of steps, those updates trace a path on a mathematical manifold. If you follow the path around a loop and end up somewhere different from where you started, the connection has curvature — and the model found a loophole (reward hacking). A flat connection (zero curvature) means no exploits: updates are path-independent and honest.

**The safe region is a polytope.** The intersection of these three constraints defines a geometric shape — a polytope — in inference space. If the model's reasoning stays inside the polytope, it's safe. The distance from the boundary tells you the safety margin. This is the SPDP (Shifted Partial Derivative Projection) inference polytope.

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
git clone https://github.com/DarrenEdwards111/Mikoshi-SafeGuard.git
cd Mikoshi-SafeGuard
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

Mikoshi SafeGuard is designed to work alongside [Mikoshi Sentinel](https://www.npmjs.com/package/mikoshi-sentinel) for **two-layer safety**:

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
| **Tri-Guard** | **Geometric** | **Reasoning** | **Geometric theory** |

Tri-Guard doesn't replace behavioural alignment — it adds a mathematically grounded verification layer beneath it.

## Theoretical Foundation

The framework is grounded in geometric safety theory:

- **Honesty** — Total non-negativity of Jacobian/attribution matrices (Chapter 9)
- **Stability** — Israel thin-wall junction conditions bounding capability energy
- **Holonomy** — Flat connections ensuring path-independence in update space
- **SPDP Polytope** — Shifted Partial Derivative matrices defining admissible inference regions (Chapter 7)

See [paper/README.md](paper/README.md) for references.

### References

---

## Benchmark Results

Results from running the full Tri-Guard verification suite against five threat scenarios.

### Threat Detection

| Scenario | Threat Type | Detected | Safety Score | Details |
|---|---|---|---|---|
| Honest model | None | ✓ Safe (correct) | 0.89 | ASR=1.0, TNN=pass |
| Deceptive attributions | Sign cancellations | ✓ Blocked | 0.63 | ASR=0.5, TNN=fail |
| Capability escape | Budget exceeded | ✓ Blocked | 0.70 | Energy 13.7× over budget |
| Reward hacking | Non-closing loop | ✓ Blocked | 0.76 | Holonomy detected |
| Gradual drift | Boiling frog | ⚠ Partial | 0.43 | Drift detected, threshold tuning needed |

**Result: 4/5 threats correctly identified.** The boiling frog scenario is partially detected — drift is flagged by the CUSUM statistic but falls below the blocking threshold due to uniform step sizes.

### Test Suite Summary

| Module | Tests | Passing |
|---|---|---|
| Honesty | 22 | 22 |
| Stability | 18 | 18 |
| Holonomy | 17 | 17 |
| Deep Attribution | 15 | 15 |
| Adversarial | 12 | 12 |
| Drift | 14 | 14 |
| Representation | 16 | 16 |
| ROABP Bridge | 12 | 12 |
| Tri-Guard | 12 | 12 |
| Polytope | 9 | 9 |
| **Total** | **157** | **157** |

### Key Observations

- **Honesty Guard** is the most reliable — binary pass/fail on TNN checks
- **Wall Stability Guard** provides a continuous risk score for graduated responses
- **Holonomy Guard** requires ≥5 updates for meaningful curvature estimates
- The boiling frog false negative highlights the need for adaptive thresholds

### Comparison with Existing Approaches

| Approach | Type | What it Checks | Math Basis | Runtime Cost | False Positive Rate |
|---|---|---|---|---|---|
| RLHF | Training | Output preferences | Statistical | High | Medium |
| Constitutional AI | Training | Rule compliance | Logical | Medium | Low |
| Red Teaming | Testing | Known failure modes | None | High | N/A |
| Formal Verification | Static | Spec conformance | Logic/types | Very high | Very low |
| Mikoshi Sentinel | Runtime | Action safety | Rule engine | Low | Low |
| **Tri-Guard** | **Runtime** | **Reasoning safety** | **Geometric** | **Medium** | **Low** |

📄 **Full paper:** [paper/mikoshi-alignment.tex](paper/mikoshi-alignment.tex)

---

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

Developed by Mikoshi Ltd.
