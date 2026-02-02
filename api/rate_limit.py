"""
In-memory rate limiter for API endpoints.

Limits requests per IP to protect against abuse and stay within
external API quotas. For multi-instance deployments, replace with
Redis-backed storage (same interface).
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple

# 100 requests per 60 seconds per IP (configurable)
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SEC = 60

# Paths that are never rate-limited (health, docs, static)
SKIP_PATHS = frozenset({
    "/health",
    "/info",
    "/docs",
    "/redoc",
    "/openapi.json",
})

# Path prefixes that are rate-limited (everything under /api)
RATE_LIMIT_PREFIX = "/api"


def _get_client_ip(request) -> str:
    """Prefer X-Forwarded-For (Render/proxy), fallback to direct client."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class InMemoryRateLimiter:
    """Sliding-window style counter per IP. Thread-safe for async (single process)."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_sec: float = RATE_LIMIT_WINDOW_SEC):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _prune(self, ip: str, now: float) -> None:
        cutoff = now - self.window_sec
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]

    def is_allowed(self, ip: str) -> Tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).
        retry_after_seconds is 0 if allowed, else seconds until a slot is free.
        """
        now = time.time()
        self._prune(ip, now)
        times = self._requests[ip]
        if len(times) < self.max_requests:
            times.append(now)
            return True, 0
        # Oldest request in window determines when we can allow next
        oldest = min(times)
        retry_after = max(0, int(self.window_sec - (now - oldest)) + 1)
        return False, retry_after


_limiter = InMemoryRateLimiter()


def check_rate_limit(request) -> Tuple[bool, int]:
    """Check if request is within rate limit. Returns (allowed, retry_after_seconds)."""
    path = request.url.path
    if path in SKIP_PATHS or not path.startswith(RATE_LIMIT_PREFIX):
        return True, 0
    ip = _get_client_ip(request)
    return _limiter.is_allowed(ip)
