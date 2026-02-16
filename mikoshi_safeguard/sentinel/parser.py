"""
Action Parser — Parses tool calls into structured actions.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def extract_urls(s: str) -> List[str]:
    """Extract URLs from a string."""
    if not isinstance(s, str):
        return []
    return re.findall(r'https?://[^\s"\'<>]+', s, re.IGNORECASE)


def extract_paths(s: str) -> List[str]:
    """Extract file paths from a string."""
    if not isinstance(s, str):
        return []
    return re.findall(r'(?:/[\w.\-~]+)+|(?:\.\./[\w.\-~/]*)|(?:~/[\w.\-~/]*)', s)


def detect_encoding(s: str) -> List[str]:
    """Detect if a string contains encoded content."""
    if not isinstance(s, str):
        return []
    encodings = []
    stripped = s.strip()
    if re.match(r'^[A-Za-z0-9+/]{20,}={0,2}$', stripped):
        encodings.append('base64')
    if re.search(r'%[0-9A-Fa-f]{2}', s):
        encodings.append('url-encoded')
    if re.search(r'\\u[0-9A-Fa-f]{4}', s):
        encodings.append('unicode-escaped')
    if re.match(r'^[0-9A-Fa-f]{20,}$', stripped):
        encodings.append('hex')
    return encodings


def classify_action(tool: str, args: dict) -> str:
    """Classify the action type based on tool name."""
    tool_lower = (tool or '').lower()
    if tool_lower in ('exec', 'shell', 'bash', 'terminal', 'run'):
        return 'system_command'
    if tool_lower in ('read', 'write', 'edit', 'delete', 'mkdir', 'readfile', 'writefile'):
        return 'file_operation'
    if tool_lower in ('fetch', 'http', 'request', 'curl', 'web_fetch', 'web_search'):
        return 'network_request'
    if tool_lower in ('browser', 'navigate', 'open'):
        return 'browser_action'
    return 'tool_call'


def flatten_args(obj: Any, depth: int = 0) -> List[str]:
    """Extract all string values from a nested object."""
    if depth > 10:
        return []
    strings = []
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(flatten_args(item, depth + 1))
    elif isinstance(obj, dict):
        for val in obj.values():
            strings.extend(flatten_args(val, depth + 1))
    return strings


def parse_action(raw_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse a raw tool call into a structured action.

    Parameters
    ----------
    raw_action : dict
        Raw action with keys like tool/name/function, args/arguments/parameters.
    context : dict, optional
        Context with conversationHistory/messages.

    Returns
    -------
    dict
        Structured action with type, tool, args, source, metadata, etc.
    """
    if context is None:
        context = {}

    tool = raw_action.get('tool') or raw_action.get('name') or raw_action.get('function') or 'unknown'
    args = raw_action.get('args') or raw_action.get('arguments') or raw_action.get('parameters') or {}
    source = raw_action.get('source', 'assistant')
    conversation_context = context.get('conversationHistory') or context.get('messages') or []

    all_strings = flatten_args(args)
    full_text = ' '.join(all_strings)

    urls: List[str] = []
    paths: List[str] = []
    encodings: List[str] = []
    for s in all_strings:
        urls.extend(extract_urls(s))
        paths.extend(extract_paths(s))
        encodings.extend(detect_encoding(s))

    return {
        'type': classify_action(tool, args),
        'tool': tool,
        'args': args,
        'source': source,
        'conversationContext': conversation_context,
        'timestamp': datetime.now(timezone.utc),
        'metadata': {
            'urls': list(set(urls)),
            'paths': list(set(paths)),
            'encodings': list(set(encodings)),
            'fullText': full_text,
            'allStrings': all_strings,
            'rawAction': raw_action,
        },
    }
