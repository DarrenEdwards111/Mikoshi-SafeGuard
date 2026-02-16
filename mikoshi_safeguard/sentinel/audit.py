"""
Audit Logger — Logs every verification decision.

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import json
import os
import random
import string
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLogger:
    """In-memory + optional file audit logger for Sentinel verdicts."""

    def __init__(self, output_file: Optional[str] = None, console: bool = False,
                 max_entries: int = 10000, verbose: bool = False):
        self.entries: List[Dict[str, Any]] = []
        self.output_file = output_file
        self.console = console
        self.max_entries = max_entries
        self.verbose = verbose

        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)

    def log(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        record = {
            'id': f'audit-{int(time.time() * 1000)}-{suffix}',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        self.entries.append(record)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        if self.console and self.verbose:
            icon = '✅' if record.get('allowed') else '🚫'
            tool = record.get('action', {}).get('tool', 'unknown')
            status = 'ALLOWED' if record.get('allowed') else 'BLOCKED'
            print(f'{icon} [Sentinel] {tool} — {status}')

        if self.output_file:
            try:
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')
            except Exception:
                pass

        return record

    def get_recent(self, count: int = 50) -> List[Dict]:
        return self.entries[-count:]

    def query(self, allowed: Optional[bool] = None, tool: Optional[str] = None,
              since: Optional[str] = None, policy: Optional[str] = None) -> List[Dict]:
        results = []
        for e in self.entries:
            if allowed is not None and e.get('allowed') != allowed:
                continue
            if tool and e.get('action', {}).get('tool') != tool:
                continue
            if since and e.get('timestamp', '') < since:
                continue
            if policy and not any(v.get('policy') == policy for v in e.get('violations', [])):
                continue
            results.append(e)
        return results

    def stats(self) -> Dict[str, Any]:
        total = len(self.entries)
        blocked = sum(1 for e in self.entries if not e.get('allowed'))
        by_policy: Dict[str, int] = {}
        for e in self.entries:
            for v in e.get('violations', []):
                p = v.get('policy', 'unknown')
                by_policy[p] = by_policy.get(p, 0) + 1
        return {'total': total, 'allowed': total - blocked, 'blocked': blocked, 'byPolicy': by_policy}

    def clear(self):
        self.entries.clear()

    def export(self) -> str:
        return json.dumps(self.entries, indent=2)
