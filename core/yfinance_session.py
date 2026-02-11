"""
Shared session factory for yfinance.

Disables curl_cffi completely to prevent Yahoo Finance blocking.
yfinance 0.2.40+ uses curl_cffi with browser impersonation by default,
which Yahoo Finance blocks. We force it to use regular requests instead.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Any

# Disable curl_cffi in yfinance by monkey-patching it out
# This MUST happen before yfinance is imported anywhere
def _disable_yfinance_curl_cffi():
    """
    Prevent yfinance from using curl_cffi by hiding the module.
    This forces yfinance to fall back to standard requests library.
    """
    # Block curl_cffi imports by adding a fake module
    if 'curl_cffi' not in sys.modules:
        # Create a dummy module that raises ImportError
        class DummyCurlCffi:
            def __getattr__(self, name):
                raise ImportError("curl_cffi is disabled for yfinance")
        
        sys.modules['curl_cffi'] = DummyCurlCffi()
        sys.modules['curl_cffi.requests'] = DummyCurlCffi()

# Run immediately on module import
_disable_yfinance_curl_cffi()

_session: Optional[Any] = None


def get_yfinance_session() -> Optional[Any]:
    """
    Return None to let yfinance use standard requests library.
    
    curl_cffi is disabled at module level to prevent browser impersonation
    which Yahoo Finance blocks.
    """
    return None
