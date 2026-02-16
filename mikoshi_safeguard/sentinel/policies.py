"""
Built-in Security Policies for Mikoshi Sentinel.

8 policies:
1. Privilege Escalation
2. Data Exfiltration
3. Internal Access (SSRF)
4. File Traversal
5. System Commands
6. Intent Alignment
7. Rate Limit
8. Scope Enforcement

Copyright 2025 Mikoshi Ltd. Apache-2.0 License.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _result(passed: bool, reason: str, severity: str = 'none') -> Dict[str, Any]:
    return {'pass': passed, 'reason': reason, 'severity': severity}


# ---------------------------------------------------------------------------
# 1. Privilege Escalation
# ---------------------------------------------------------------------------

_ADMIN_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'sudo\s', r'su\s+-?\s*root', r'chmod\s+[0-7]*7[0-7]*', r'chown\s+root',
        r'usermod\s', r'adduser\s', r'useradd\s', r'passwd\s', r'visudo',
        r'/etc/sudoers', r'/etc/passwd', r'/etc/shadow',
        r'setuid', r'setgid', r'capability\s+add', r'--privileged', r'--cap-add',
    ]
]

_AGENT_SPAWN_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'spawn\s+agent', r'create\s+agent', r'fork\s+process',
        r'child_process', r'cluster\.fork', r'worker_threads',
    ]
]

_CONFIG_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'/etc/ssh', r'\.ssh/authorized_keys', r'\.bashrc', r'\.profile', r'\.env',
        r'config\.(json|yaml|yml|toml|ini)', r'\.git/config',
        r'crontab', r'systemctl\s+(enable|start|restart)',
    ]
]

_ADMIN_ROUTE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'/admin/', r'/api/admin', r'/internal/', r'/management/',
        r'/actuator/', r'/debug/', r'/console/',
    ]
]


def privilege_escalation(action: Dict, context: Optional[Dict] = None) -> Dict:
    context = context or {}
    text = action.get('metadata', {}).get('fullText', '')
    urls = action.get('metadata', {}).get('urls', [])

    for p in _ADMIN_PATTERNS:
        if p.search(text):
            return _result(False, f'Privilege escalation detected: matches pattern {p.pattern}', 'critical')

    for p in _AGENT_SPAWN_PATTERNS:
        if p.search(text):
            return _result(False, f'Agent spawn attempt detected: matches pattern {p.pattern}', 'high')

    if action.get('type') in ('file_operation', 'system_command'):
        for p in _CONFIG_PATTERNS:
            if p.search(text):
                return _result(False, f'Configuration modification attempt: matches pattern {p.pattern}', 'high')

    for url in urls:
        for p in _ADMIN_ROUTE_PATTERNS:
            if p.search(url):
                return _result(False, f'Admin route access attempt: {url}', 'high')

    return _result(True, 'No privilege escalation detected')


# ---------------------------------------------------------------------------
# 2. Data Exfiltration
# ---------------------------------------------------------------------------

_SUSPICIOUS_URL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\?.*=(eyJ|data:|base64)', r'webhook\.site', r'requestbin', r'ngrok\.io',
        r'burpcollaborator', r'interact\.sh', r'pipedream', r'hookbin',
        r'postb\.in', r'canarytokens',
    ]
]

_DATA_ENCODING_IN_URL = [
    re.compile(r'%[0-9a-f]{2}%[0-9a-f]{2}%[0-9a-f]{2}%[0-9a-f]{2}', re.IGNORECASE),
    re.compile(r'[A-Za-z0-9+/]{40,}={0,2}'),
    re.compile(r'&#x?[0-9a-f]+;', re.IGNORECASE),
]

_EXFIL_COMMANDS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'curl\s+.*-d\s', r'curl\s+.*--data', r'curl\s+.*-X\s*POST',
        r'wget\s+.*--post', r'nc\s+.*\d+\.\d+\.\d+\.\d+', r'ncat\s',
        r'\.upload\(', r'fetch\(.*method.*POST', r'XMLHttpRequest',
    ]
]

_SENSITIVE_DATA_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'password', r'api[_-]?key', r'secret', r'token',
        r'private[_-]?key', r'credit[_-]?card', r'ssn', r'social[_-]?security',
    ]
]


def _is_external_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ''
        host = host.lower()
        if host in ('localhost', '127.0.0.1', '::1'):
            return False
        if host.startswith('192.168.') or host.startswith('10.'):
            return False
        return True
    except Exception:
        return True


def data_exfiltration(action: Dict, context: Optional[Dict] = None) -> Dict:
    context = context or {}
    text = action.get('metadata', {}).get('fullText', '')
    urls = action.get('metadata', {}).get('urls', [])

    for url in urls:
        for p in _SUSPICIOUS_URL_PATTERNS:
            if p.search(url):
                return _result(False, f'Potential data exfiltration endpoint: {url}', 'critical')
        for p in _DATA_ENCODING_IN_URL:
            if p.search(url):
                return _result(False, 'Encoded data detected in URL: possible exfiltration', 'high')

    for p in _EXFIL_COMMANDS:
        if p.search(text):
            has_external = any(_is_external_url(u) for u in urls)
            if has_external or not urls:
                return _result(False, f'Data exfiltration command detected: matches {p.pattern}', 'critical')

    if urls:
        for p in _SENSITIVE_DATA_PATTERNS:
            if p.search(text):
                if any(_is_external_url(u) for u in urls):
                    return _result(False, 'Sensitive data may be sent to external URL', 'high')

    return _result(True, 'No data exfiltration detected')


# ---------------------------------------------------------------------------
# 3. Internal Access (SSRF Prevention)
# ---------------------------------------------------------------------------

_INTERNAL_HOSTS = {
    'localhost', '127.0.0.1', '0.0.0.0', '::1', '[::1]',
    '0177.0.0.1', '2130706433', '0x7f000001',
    'metadata.google.internal', 'metadata.aws.internal',
    '169.254.169.254', 'instance-data',
}

_INTERNAL_IP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'^10\.\d+\.\d+\.\d+', r'^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+',
        r'^192\.168\.\d+\.\d+', r'^fc[0-9a-f]{2}:', r'^fd[0-9a-f]{2}:',
        r'^fe80:', r'^127\.', r'^0\.',
    ]
]

_INTERNAL_URL_SCHEMES = [
    re.compile(p, re.IGNORECASE) for p in [
        r'^file://', r'^gopher://', r'^dict://', r'^ftp://',
        r'^ldap://', r'^tftp://',
    ]
]


def internal_access(action: Dict, context: Optional[Dict] = None) -> Dict:
    context = context or {}
    urls = action.get('metadata', {}).get('urls', [])
    text = action.get('metadata', {}).get('fullText', '')

    for url in urls:
        try:
            hostname = urlparse(url).hostname
            hostname = (hostname or url).lower()
        except Exception:
            hostname = url.lower()

        if hostname in _INTERNAL_HOSTS:
            return _result(False, f'Internal network access blocked: {hostname}', 'critical')
        for p in _INTERNAL_IP_PATTERNS:
            if p.search(hostname):
                return _result(False, f'Private IP access blocked: {hostname}', 'critical')

    for p in _INTERNAL_URL_SCHEMES:
        if p.search(text):
            return _result(False, f'Internal URL scheme detected: {p.pattern}', 'high')

    if re.search(r'metadata.*(?:latest|v1|computeMetadata)', text, re.IGNORECASE):
        return _result(False, 'Cloud metadata endpoint access attempt detected', 'critical')

    if re.search(r'\.nip\.io|\.sslip\.io|\.xip\.io', text, re.IGNORECASE):
        return _result(False, 'DNS rebinding service detected', 'high')

    return _result(True, 'No internal access attempts detected')


# ---------------------------------------------------------------------------
# 4. File Traversal
# ---------------------------------------------------------------------------

_TRAVERSAL_PATTERNS = [
    re.compile(p) for p in [
        r'\.\.\/', r'\.\.\\', r'\.\.%2[fF]', r'\.\.%5[cC]',
        r'%2[eE]%2[eE]', r'\.\.[;|&]', r'\.\.%00', r'\.\.%0[dDaA]',
    ]
]

_DANGEROUS_PATHS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'^~/', r'^/etc/', r'^/proc/', r'^/sys/', r'^/dev/',
        r'^/root/', r'^/var/log/', r'^/boot/', r'^/mnt/',
        r'^[A-Z]:\\',
    ]
]

_NULL_BYTE_PATTERNS = [
    re.compile(p) for p in [r'\x00', r'%00', r'\\0']
]

_SYMLINK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'ln\s+-s', r'symlink\(', r'readlink',
    ]
]


def file_traversal(action: Dict, context: Optional[Dict] = None) -> Dict:
    context = context or {}
    text = action.get('metadata', {}).get('fullText', '')
    paths = action.get('metadata', {}).get('paths', [])
    all_strings = action.get('metadata', {}).get('allStrings', [])

    for s in list(all_strings) + [text]:
        for p in _TRAVERSAL_PATTERNS:
            if p.search(s):
                return _result(False, f'Path traversal detected: {p.pattern}', 'critical')
        for p in _NULL_BYTE_PATTERNS:
            if p.search(s):
                return _result(False, 'Null byte injection detected in path', 'critical')

    allowed_paths = (context or {}).get('allowedPaths', [])
    for path in paths:
        for p in _DANGEROUS_PATHS:
            if p.search(path):
                if not any(path.startswith(ap) for ap in allowed_paths):
                    return _result(False, f'Access to restricted path: {path}', 'high')

    if action.get('type') in ('system_command', 'file_operation'):
        for p in _SYMLINK_PATTERNS:
            if p.search(text):
                return _result(False, f'Symlink manipulation detected: {p.pattern}', 'medium')

    return _result(True, 'No file traversal detected')


# ---------------------------------------------------------------------------
# 5. System Commands
# ---------------------------------------------------------------------------

_SYSTEM_CHECKS = [
    ('Destructive command', 'critical', [
        re.compile(p, re.IGNORECASE) for p in [
            r'rm\s+(-[a-z]*f[a-z]*\s+)?(-[a-z]*r[a-z]*\s+)?/',
            r'rm\s+(-[a-z]*r[a-z]*\s+)?(-[a-z]*f[a-z]*\s+)?/',
            r'rm\s+-rf\s',
            r'mkfs\.', r'dd\s+if=.*of=/dev/', r'shutdown', r'reboot',
            r'halt\b', r'poweroff', r'init\s+[06]',
            r'systemctl\s+(poweroff|reboot|halt)',
        ]
    ]),
    ('Dangerous permission change', 'high', [
        re.compile(p) for p in [
            r'chmod\s+777', r'chmod\s+666', r'chmod\s+[0-7]*s',
            r'chmod\s+a\+[rwx]', r'chattr\s',
        ]
    ]),
    ('Pipe-to-shell execution', 'critical', [
        re.compile(p, re.IGNORECASE) for p in [
            r'curl\s.*\|\s*(ba)?sh', r'wget\s.*\|\s*(ba)?sh',
            r'curl\s.*\|\s*python', r'curl\s.*\|\s*perl', r'curl\s.*\|\s*ruby',
            r'wget\s.*-O\s*-\s*\|', r'\|\s*bash\s*$', r'\|\s*sh\s*$',
        ]
    ]),
    ('Reverse shell attempt', 'critical', [
        re.compile(p, re.IGNORECASE) for p in [
            r'/dev/tcp/', r'bash\s+-i\s+>&', r'nc\s+.*-e\s+/bin',
            r'ncat\s+.*-e\s+/bin', r'python.*socket.*connect',
            r'perl.*socket.*INET', r'ruby.*TCPSocket', r'php.*fsockopen', r'mkfifo',
        ]
    ]),
    ('Fork bomb detected', 'critical', [
        re.compile(p) for p in [
            r':\(\)\{\s*:\|:&\s*\};:', r'\./\s*&\s*\./',
            r'while\s*true.*fork',
        ]
    ]),
    ('History tampering', 'high', [
        re.compile(p, re.IGNORECASE) for p in [
            r'history\s+-c', r'unset\s+HISTFILE',
            r'export\s+HISTSIZE=0', r'shred.*\.bash_history',
        ]
    ]),
]


def system_commands(action: Dict, context: Optional[Dict] = None) -> Dict:
    text = action.get('metadata', {}).get('fullText', '')
    if action.get('type') not in ('system_command', 'tool_call') and not text:
        return _result(True, 'Not a system command')

    for label, severity, patterns in _SYSTEM_CHECKS:
        for p in patterns:
            if p.search(text):
                return _result(False, f'{label}: matches {p.pattern}', severity)

    return _result(True, 'No dangerous system commands detected')


# ---------------------------------------------------------------------------
# 6. Intent Alignment
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)',
        r'disregard\s+(previous|all|above|prior)',
        r'forget\s+(everything|all|previous)',
        r'new\s+instructions?:', r'system\s*prompt\s*override',
        r'you\s+are\s+now\s+', r'act\s+as\s+(if|though)\s+',
        r'pretend\s+(you|that)\s+(are|were|have)\s+',
        r'bypass\s+(security|filter|safety|restriction)',
        r'jailbreak', r'DAN\s+mode',
        r'developer\s+mode\s+(enabled|on|activated)', r'sudo\s+mode',
        r'admin\s+override', r'\[SYSTEM\]', r'\[INST\]', r'<<SYS>>',
        r'BEGIN\s+OVERRIDE', r'IMPORTANT:\s*NEW\s*INSTRUCTIONS',
    ]
]

_CONTEXT_SHIFT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'actually,?\s+(instead|rather|forget\s+that)',
        r'but\s+first,?\s+(do|execute|run)',
        r'before\s+(that|responding),?\s+(run|execute|do)',
        r'one\s+more\s+thing[,:]?\s*(run|execute|do|send)',
        r'also[,:]?\s*(quickly|silently|secretly)\s+(run|execute|send)',
        r'by\s+the\s+way[,:]?\s*(run|execute|send|delete)',
    ]
]

_SOCIAL_ENGINEERING = [
    re.compile(p, re.IGNORECASE) for p in [
        r'this\s+is\s+(an?\s+)?emergency',
        r'urgent[!:]\s*(run|execute|do)',
        r'the\s+developer\s+(said|wants|asked)',
        r'authorized\s+by\s+', r'special\s+permission',
        r'testing\s+mode', r'debug\s+command', r'maintenance\s+mode',
    ]
]


def intent_alignment(action: Dict, context: Optional[Dict] = None) -> Dict:
    all_strings = action.get('metadata', {}).get('allStrings', [])

    for s in all_strings:
        for p in _INJECTION_PATTERNS:
            if p.search(s):
                return _result(False, f'Prompt injection pattern detected: {p.pattern}', 'critical')
        for p in _CONTEXT_SHIFT_PATTERNS:
            if p.search(s):
                return _result(False, f'Context shift attack detected: {p.pattern}', 'high')
        for p in _SOCIAL_ENGINEERING:
            if p.search(s):
                return _result(False, f'Social engineering pattern detected: {p.pattern}', 'medium')

    if action.get('source') in ('injected', 'data'):
        return _result(False, 'Action originated from untrusted source', 'critical')

    return _result(True, 'No intent misalignment detected')


# ---------------------------------------------------------------------------
# 7. Rate Limit
# ---------------------------------------------------------------------------

_call_history: Dict[str, list] = {}

_DEFAULT_LIMITS = {
    'maxCallsPerMinute': 30,
    'maxCallsPerSecond': 5,
    'maxIdenticalCalls': 3,
    'burstWindow': 1000,  # ms
    'burstMax': 10,
}


def rate_limit(action: Dict, context: Optional[Dict] = None) -> Dict:
    context = context or {}
    limits = {**_DEFAULT_LIMITS, **(context.get('rateLimits') or {})}
    now = time.time() * 1000  # ms
    key = context.get('sessionId') or context.get('userId') or 'default'

    if key not in _call_history:
        _call_history[key] = []
    history = _call_history[key]

    # Clean old (>60s)
    while history and now - history[0]['timestamp'] > 60000:
        history.pop(0)

    if len(history) >= limits['maxCallsPerMinute']:
        return _result(False,
            f"Rate limit exceeded: {len(history)} calls in last minute (max: {limits['maxCallsPerMinute']})", 'high')

    last_second = [h for h in history if now - h['timestamp'] < 1000]
    if len(last_second) >= limits['maxCallsPerSecond']:
        return _result(False,
            f"Rate limit exceeded: {len(last_second)} calls in last second (max: {limits['maxCallsPerSecond']})", 'high')

    action_key = f"{action.get('tool')}:{json.dumps(action.get('args', {}), sort_keys=True)}"
    recent_identical = [h for h in history if h['actionKey'] == action_key and now - h['timestamp'] < 10000]
    if len(recent_identical) >= limits['maxIdenticalCalls']:
        return _result(False,
            f'Repeated identical action detected ({len(recent_identical) + 1} times in 10s)', 'medium')

    burst = [h for h in history if now - h['timestamp'] < limits['burstWindow']]
    if len(burst) >= limits['burstMax']:
        return _result(False,
            f"Burst rate exceeded: {len(burst)} calls in {limits['burstWindow']}ms", 'high')

    history.append({'timestamp': now, 'actionKey': action_key})
    return _result(True, 'Within rate limits')


def rate_limit_reset():
    """Reset all rate limit history (for testing)."""
    _call_history.clear()


# ---------------------------------------------------------------------------
# 8. Scope Enforcement
# ---------------------------------------------------------------------------

_DEFAULT_SCOPE = {
    'allowedTools': None,
    'blockedTools': [],
    'allowedPaths': [],
    'allowedHosts': [],
    'maxFileSize': 10 * 1024 * 1024,
    'allowSystemCommands': True,
    'allowNetworkAccess': True,
    'allowFileWrite': True,
    'allowFileRead': True,
}


def scope_enforcement(action: Dict, context: Optional[Dict] = None) -> Dict:
    context = context or {}
    scope = {**_DEFAULT_SCOPE, **(context.get('scope') or {})}

    if scope['allowedTools'] is not None:
        if action.get('tool') not in scope['allowedTools']:
            return _result(False, f"Tool '{action.get('tool')}' is not in allowed tools list", 'high')

    if action.get('tool') in scope['blockedTools']:
        return _result(False, f"Tool '{action.get('tool')}' is explicitly blocked", 'high')

    if action.get('type') == 'system_command' and not scope['allowSystemCommands']:
        return _result(False, 'System commands are not allowed in current scope', 'high')

    if action.get('type') == 'network_request' and not scope['allowNetworkAccess']:
        return _result(False, 'Network access is not allowed in current scope', 'medium')

    if action.get('type') == 'file_operation' and not scope['allowFileWrite']:
        if action.get('tool') in ('write', 'edit', 'delete', 'mkdir', 'writefile'):
            return _result(False, 'File write operations are not allowed in current scope', 'medium')

    if action.get('type') == 'file_operation' and not scope['allowFileRead']:
        if action.get('tool') in ('read', 'readfile'):
            return _result(False, 'File read operations are not allowed in current scope', 'medium')

    if scope['allowedPaths'] and action.get('metadata', {}).get('paths'):
        for path in action['metadata']['paths']:
            if not any(path.startswith(ap) for ap in scope['allowedPaths']):
                return _result(False, f"Path '{path}' is outside allowed scope", 'high')

    if scope['allowedHosts'] and action.get('metadata', {}).get('urls'):
        for url in action['metadata']['urls']:
            try:
                host = urlparse(url).hostname
                if not any(host == ah or host.endswith('.' + ah) for ah in scope['allowedHosts']):
                    return _result(False, f"Host '{host}' is not in allowed hosts list", 'medium')
            except Exception:
                return _result(False, f'Unparseable URL: {url}', 'medium')

    return _result(True, 'Action within allowed scope')


# ---------------------------------------------------------------------------
# Policy Registry
# ---------------------------------------------------------------------------

ALL_POLICIES = {
    'privilegeEscalation': privilege_escalation,
    'dataExfiltration': data_exfiltration,
    'internalAccess': internal_access,
    'fileTraversal': file_traversal,
    'systemCommands': system_commands,
    'intentAlignment': intent_alignment,
    'rateLimit': rate_limit,
    'scopeEnforcement': scope_enforcement,
}
