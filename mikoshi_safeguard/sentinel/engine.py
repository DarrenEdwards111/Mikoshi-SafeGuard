"""
Sentinel Engine — Main policy engine for action verification.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from mikoshi_safeguard.sentinel.audit import AuditLogger
from mikoshi_safeguard.sentinel.parser import parse_action
from mikoshi_safeguard.sentinel.policies import ALL_POLICIES
from mikoshi_safeguard.sentinel.verifier import IntentVerifier


class Sentinel:
    """Deterministic action verification engine for LLM agent security.

    Parameters
    ----------
    enable_intent_verification : bool
        Enable heuristic/LLM intent checking (default True).
    llm_fn : callable, optional
        Async function for LLM calls: (prompt) => string.
    intent_threshold : float
        Minimum intent confidence (0-1), default 0.5.
    audit : dict
        Audit logger options.
    use_builtin_policies : bool
        Load all 8 built-in policies (default True).
    """

    def __init__(self, enable_intent_verification: bool = True,
                 llm_fn: Optional[Callable] = None,
                 intent_threshold: float = 0.5,
                 audit: Optional[Dict] = None,
                 use_builtin_policies: bool = True,
                 **kwargs):
        self.policies: List[Dict[str, Any]] = []
        self.audit_log = AuditLogger(**(audit or {}))
        self.config = {
            'enable_intent_verification': enable_intent_verification,
            'llm_fn': llm_fn,
            'intent_threshold': intent_threshold,
            **kwargs,
        }

        if enable_intent_verification:
            self.intent_verifier: Optional[IntentVerifier] = IntentVerifier(
                llm_fn=llm_fn, threshold=intent_threshold)
        else:
            self.intent_verifier = None

        if use_builtin_policies:
            for name, fn in ALL_POLICIES.items():
                self.add_policy(name, fn)

    def add_policy(self, name: str, check_fn: Callable) -> 'Sentinel':
        if not callable(check_fn):
            raise TypeError(f"Policy '{name}' must be callable")
        self.policies.append({'name': name, 'check': check_fn})
        return self

    def remove_policy(self, name: str) -> 'Sentinel':
        self.policies = [p for p in self.policies if p['name'] != name]
        return self

    async def verify(self, raw_action: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main verification pipeline.

        Returns dict with: allowed, confidence, violations, intent, policyResults, action, elapsed.
        """
        context = context or {}
        start = time.perf_counter()

        action = parse_action(raw_action, context)

        violations = []
        policy_results = []

        for policy in self.policies:
            try:
                result = policy['check'](action, context)
                policy_results.append({'policy': policy['name'], **result})
                if not result.get('pass'):
                    violations.append({
                        'policy': policy['name'],
                        'reason': result.get('reason', ''),
                        'severity': result.get('severity', 'medium'),
                    })
            except Exception as e:
                violations.append({
                    'policy': policy['name'],
                    'reason': f'Policy error: {e}',
                    'severity': 'high',
                })

        intent_result = None
        if self.intent_verifier and not violations:
            try:
                intent_result = await self.intent_verifier.verify(
                    action, action.get('conversationContext', []))
            except Exception as e:
                intent_result = {
                    'confidence': 0.5, 'aligned': True, 'method': 'error',
                    'reason': f'Intent verification error: {e}',
                }

        allowed = len(violations) == 0
        has_critical = any(v.get('severity') == 'critical' for v in violations)

        if allowed:
            confidence = intent_result['confidence'] if intent_result else 1.0
        elif has_critical:
            confidence = 0.0
        else:
            confidence = 0.2

        elapsed = (time.perf_counter() - start) * 1000

        verdict = {
            'allowed': allowed,
            'confidence': confidence,
            'violations': violations,
            'intent': intent_result,
            'policyResults': policy_results,
            'action': {
                'type': action['type'],
                'tool': action['tool'],
                'args': action['args'],
                'source': action['source'],
                'timestamp': action['timestamp'].isoformat() if hasattr(action['timestamp'], 'isoformat') else str(action['timestamp']),
            },
            'elapsed': f'{elapsed:.2f}ms',
        }

        self.audit_log.log(verdict)
        return verdict

    def stats(self) -> Dict[str, Any]:
        return self.audit_log.stats()
