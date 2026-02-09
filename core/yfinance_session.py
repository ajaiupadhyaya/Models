"""
Shared session factory for yfinance.

Newer yfinance versions require a curl_cffi session when a session is provided.
If curl_cffi is unavailable, return None and let yfinance manage its own session.
"""
from __future__ import annotations

import os
from typing import Optional, Any

_session: Optional[Any] = None

# Use a conservative impersonation target that is widely supported.
DEFAULT_IMPERSONATE = os.getenv("YFINANCE_IMPERSONATE", "chrome110")


def get_yfinance_session() -> Optional[Any]:
    """Return a curl_cffi session for yfinance if available; otherwise None."""
    global _session
    if _session is not None:
        return _session

    try:
        from curl_cffi import requests as curl_requests

        _session = curl_requests.Session(impersonate=DEFAULT_IMPERSONATE)
        return _session
    except Exception:
        # Fallback: yfinance will create its own session internally.
        return None
