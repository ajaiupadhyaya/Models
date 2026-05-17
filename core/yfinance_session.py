"""
Neutralize curl_cffi browser impersonation for yfinance.

yfinance >= 0.2.40 hard-imports curl_cffi (`from curl_cffi import requests`)
and constructs sessions as `curl_cffi.requests.Session(impersonate="chrome")`.
Yahoo Finance has been blocking the impersonated User-Agents intermittently,
which would surface as 401/403 errors deep in our code paths.

Strategy: leave the curl_cffi *import* untouched (so yfinance imports cleanly),
but replace `curl_cffi.requests.Session` with a thin shim that delegates to a
plain `requests.Session` and silently swallows the `impersonate=` kwarg.

This module must be imported before any `import yfinance`. `api/main.py` imports
it as line 31, which runs before the first yfinance import in the router tree.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

import requests as _requests


class _PlainSession(_requests.Session):
    """requests.Session that accepts and ignores curl_cffi's impersonate=/etc. kwargs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        kwargs.pop("impersonate", None)
        kwargs.pop("default_headers", None)
        super().__init__()
        # A realistic UA helps avoid generic anti-bot heuristics.
        self.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
        )


def _patch_curl_cffi() -> None:
    """Replace curl_cffi.requests.Session with our shim (idempotent)."""
    try:
        import curl_cffi  # noqa: F401
        from curl_cffi import requests as _curl_requests
    except Exception:
        # curl_cffi not installed — yfinance may itself fail to import, but
        # that is a separate concern; nothing to patch here.
        return

    _curl_requests.Session = _PlainSession  # type: ignore[attr-defined]
    if hasattr(_curl_requests, "AsyncSession"):
        # Best-effort: don't impersonate in async path either.
        _curl_requests.AsyncSession = _PlainSession  # type: ignore[attr-defined]


_patch_curl_cffi()


def get_yfinance_session() -> Optional[_requests.Session]:
    """Return a plain requests.Session for callers that want to pass one explicitly."""
    return _PlainSession()
