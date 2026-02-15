# Theoretical Foundation

## N-Frame Theory

The Tri-Guard safety framework is grounded in N-Frame physics (Edwards, 2023, 2024, 2025),
which provides a geometric foundation for AI alignment.

### Key Results

1. **Honesty Guard** — From Chapter 9: Total non-negativity (TNN) of attribution matrices
   ensures faithful representation of model reasoning.

2. **Wall Stability Guard** — From Israel thin-wall junction conditions: capability energy
   is bounded by a domain wall with computable tension.

3. **Holonomy Guard** — From flat connections in fibre bundles: trivial holonomy around
   loops in update space rules out reward hacking.

### SPDP Framework

The Shifted Partial Derivative Polytope (SPDP) framework from Chapter 7 provides:
- Polynomial complexity bounds via ROABP width
- Admissible inference regions as convex polytopes
- Observer-dependent rank measures

### References

- Edwards, D. (2023). *N-Frame: A Geometric Framework for Physics and Information*.
- Edwards, D. (2024). *Extensions to N-Frame Theory: Applications in AI Safety*.
- Edwards, D. (2025). *Tri-Guard: Geometric Safety Verification for AI Systems*.

### Citation

```bibtex
@software{mikoshi_alignment,
  title={Mikoshi AI Alignment: Geometric Safety Verification},
  author={{Mikoshi Ltd}},
  year={2025},
  url={https://github.com/DarrenEdwards111/Mikoshi-AI-Alignment}
}
```
