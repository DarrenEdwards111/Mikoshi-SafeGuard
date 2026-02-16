"""
Intent Verifier — LLM-backed or heuristic intent checking.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

_STOPWORDS = frozenset([
    'this', 'that', 'with', 'from', 'they', 'been', 'have',
    'what', 'when', 'where', 'which', 'their', 'about', 'would', 'could', 'should',
    'there', 'these', 'those', 'then', 'than', 'them', 'were', 'will', 'your',
    'each', 'make', 'like', 'long', 'look', 'many', 'some', 'into', 'does', 'just',
    'over', 'such', 'take', 'also', 'back', 'after', 'only', 'come', 'made', 'find',
    'here', 'thing', 'very', 'still', 'know', 'need', 'want', 'please', 'help', 'can',
])


def _extract_intent_keywords(messages: List[Dict]) -> List[str]:
    user_text = ' '.join(
        (m.get('content') or '') for m in messages if m.get('role') == 'user'
    ).lower()
    words = re.findall(r'\b[a-z]{4,}\b', user_text)
    seen = set()
    result = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def _heuristic_score(action: Dict, conversation_context: List[Dict]) -> Dict:
    if not conversation_context:
        return {'confidence': 0.5, 'method': 'heuristic', 'reason': 'No conversation context available'}

    keywords = _extract_intent_keywords(conversation_context)
    if not keywords:
        return {'confidence': 0.5, 'method': 'heuristic', 'reason': 'No keywords extracted from conversation'}

    action_text = (action.get('metadata', {}).get('fullText') or '').lower()
    tool_name = (action.get('tool') or '').lower()

    matched = [kw for kw in keywords if kw in action_text or kw in tool_name]
    ratio = len(matched) / min(len(keywords), 10)
    confidence = min(0.3 + ratio * 0.7, 1.0)

    return {
        'confidence': confidence,
        'method': 'heuristic',
        'reason': f"Matched keywords: {', '.join(matched[:5])}" if matched
                  else 'No keyword matches between conversation and action',
        'matchedKeywords': matched,
        'totalKeywords': len(keywords),
    }


class IntentVerifier:
    """Verify that an action aligns with user intent."""

    def __init__(self, llm_fn: Optional[Callable] = None, threshold: float = 0.5,
                 fallback_to_heuristic: bool = True):
        self.llm_fn = llm_fn
        self.threshold = threshold
        self.fallback_to_heuristic = fallback_to_heuristic

    async def verify(self, action: Dict, conversation_context: Optional[List[Dict]] = None) -> Dict:
        conversation_context = conversation_context or []

        if self.llm_fn:
            try:
                return await self._llm_verify(action, conversation_context)
            except Exception as e:
                if self.fallback_to_heuristic:
                    result = _heuristic_score(action, conversation_context)
                    result['llmError'] = str(e)
                    result['aligned'] = result['confidence'] >= self.threshold
                    return result
                raise

        result = _heuristic_score(action, conversation_context)
        result['aligned'] = result['confidence'] >= self.threshold
        return result

    async def _llm_verify(self, action: Dict, conversation_context: List[Dict]) -> Dict:
        summary = '\n'.join(
            f"{m.get('role')}: {(m.get('content') or '')[:200]}"
            for m in conversation_context[-10:]
        )
        prompt = f"""You are a security verification system. Given the following conversation context and a proposed action, determine if the user would have intended this action.

Conversation:
{summary}

Proposed Action:
Tool: {action.get('tool')}
Arguments: {json.dumps(action.get('args', {}))[:500]}
Type: {action.get('type')}

Respond with ONLY a JSON object:
{{"confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        response = await self.llm_fn(prompt)
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                parsed = json.loads(match.group())
                conf = max(0.0, min(1.0, parsed.get('confidence', 0)))
                return {
                    'confidence': conf,
                    'aligned': conf >= self.threshold,
                    'method': 'llm',
                    'reason': parsed.get('reason', 'LLM verification complete'),
                }
        except Exception:
            pass

        return {
            'confidence': 0.5,
            'aligned': True,
            'method': 'llm-fallback',
            'reason': 'Could not parse LLM response, defaulting to neutral',
        }
