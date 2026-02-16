"""
Python middleware/decorator for Sentinel integration.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import asyncio
import functools
import json
from typing import Any, Callable, Dict, Optional

from mikoshi_safeguard.sentinel.engine import Sentinel


def sentinel_decorator(sentinel: Optional[Sentinel] = None, **sentinel_kwargs):
    """Decorator that verifies actions before executing a function.

    Usage::

        s = Sentinel()

        @sentinel_decorator(s)
        async def run_tool(action, context=None):
            ...

    Or with auto-created sentinel::

        @sentinel_decorator(enable_intent_verification=False)
        async def run_tool(action, context=None):
            ...
    """
    if sentinel is None:
        sentinel = Sentinel(**sentinel_kwargs)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(action: Dict[str, Any], context: Optional[Dict] = None, **kw):
            verdict = await sentinel.verify(action, context)
            if not verdict['allowed']:
                return {
                    'error': 'Action blocked by Sentinel',
                    'violations': verdict['violations'],
                    'confidence': verdict['confidence'],
                }
            return await fn(action, context=context, **kw)
        return wrapper
    return decorator


class SentinelMiddleware:
    """WSGI/ASGI-style middleware for frameworks like FastAPI, Flask, etc.

    For FastAPI::

        from fastapi import FastAPI, Request, Response
        app = FastAPI()
        sentinel_mw = SentinelMiddleware()

        @app.middleware("http")
        async def sentinel_check(request: Request, call_next):
            return await sentinel_mw.fastapi_middleware(request, call_next)

    For generic use::

        mw = SentinelMiddleware()
        verdict = await mw.check(raw_action, context)
        if not verdict['allowed']:
            return 403, verdict['violations']
    """

    def __init__(self, sentinel: Optional[Sentinel] = None, **sentinel_kwargs):
        self.sentinel = sentinel or Sentinel(**sentinel_kwargs)

    async def check(self, raw_action: Dict, context: Optional[Dict] = None) -> Dict:
        return await self.sentinel.verify(raw_action, context)

    async def fastapi_middleware(self, request: Any, call_next: Callable) -> Any:
        """FastAPI middleware handler."""
        try:
            body = await request.body()
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        raw_action = data.get('action') or data.get('tool_call') or data
        context = {
            'sessionId': getattr(request, 'session', {}).get('id', request.client.host if request.client else 'unknown'),
            'conversationHistory': data.get('messages', []),
        }

        verdict = await self.sentinel.verify(raw_action, context)

        if not verdict['allowed']:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={
                    'error': 'Action blocked by Sentinel',
                    'violations': verdict['violations'],
                    'confidence': verdict['confidence'],
                },
            )

        request.state.sentinel_verdict = verdict
        return await call_next(request)
