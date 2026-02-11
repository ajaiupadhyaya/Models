"""
Shared session factory for yfinance.

Newer yfinance versions can work with or without a custom session.
We disable curl_cffi impersonation to avoid browser detection blocking issues.
Instead, we set a proper User-Agent header via yfinance's built-in mechanism.
"""
from __future__ import annotations

import os
from typing import Optional, Any

_session: Optional[Any] = None


def get_yfinance_session() -> Optional[Any]:
    """
    Return None to let yfinance handle sessions internally.
    
    This avoids curl_cffi blocking from Yahoo Finance.
    yfinance will use its own session management which is more reliable.
    """
    return None
