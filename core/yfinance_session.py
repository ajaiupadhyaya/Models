"""
yfinance session helper (yfinance >= 1.0).

yfinance 1.x requires curl_cffi sessions internally. Do not patch curl_cffi or
pass a custom requests.Session — let yfinance create its own session.

This module is imported early from api/main.py so any legacy callers that
imported it before yfinance still get a stable no-op shim.
"""
from __future__ import annotations

from typing import Optional


def get_yfinance_session() -> Optional[None]:
    """Return None so yfinance uses its default curl_cffi session."""
    return None
