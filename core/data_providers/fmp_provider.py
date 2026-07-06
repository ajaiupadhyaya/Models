"""
Financial Modeling Prep (FMP) data provider.

Supports: company profile, income statement, balance sheet, cash flow, peers.
Docs: https://site.financialmodelingprep.com/developer/docs
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import time

import requests

from .base import DataProvider, AssetType

logger = logging.getLogger(__name__)


class FMPProvider(DataProvider):
    """Financial Modeling Prep API provider for fundamentals."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("fmp", api_key or os.getenv("FMP_API_KEY"))
        self.rate_limit = 300  # free tier ~250/day, premium higher
        if not self.api_key:
            logger.warning("FMP_API_KEY not found; FMP provider disabled")

    def supports_asset_type(self, asset_type: AssetType) -> bool:
        return asset_type == AssetType.EQUITY

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day",
    ) -> List:
        """FMP can provide historical price via /historical-price-full; not implemented here (use yfinance/Polygon)."""
        return []

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> requests.Response:
        url = f"{self.BASE_URL}{path}"
        params = dict(params or {})
        if self.api_key:
            params["apikey"] = self.api_key
        for attempt in range(self.max_retries):
            try:
                r = self.session.request(method, url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                return r
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2 ** attempt))
        return self.session.request(method, url, params=params, timeout=self.timeout)

    def fetch_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch company profile (name, sector, industry, market cap, description)."""
        if not self.api_key:
            return None
        r = self._request("get", f"/profile/{symbol.upper()}")
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def fetch_income_statement(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        """period: annual | quarterly."""
        if not self.api_key:
            return []
        r = self._request("get", f"/income-statement/{symbol.upper()}", {"period": period, "limit": limit})
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []

    def fetch_balance_sheet(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        r = self._request("get", f"/balance-sheet-statement/{symbol.upper()}", {"period": period, "limit": limit})
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []

    def fetch_cash_flow(self, symbol: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        r = self._request("get", f"/cash-flow-statement/{symbol.upper()}", {"period": period, "limit": limit})
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []

    def fetch_peers(self, symbol: str) -> List[str]:
        """Return list of peer ticker symbols."""
        if not self.api_key:
            return []
        r = self._request("get", "/stock_peers", {"symbol": symbol.upper()})
        if r.status_code != 200:
            return []
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            peers = data[0].get("peersList") or data[0].get("peers") or []
            return [p for p in peers if isinstance(p, str)]
        return []

    def fetch_key_metrics_ttm(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Key metrics (P/E, EV/EBITDA, etc.) TTM."""
        if not self.api_key:
            return None
        r = self._request("get", f"/key-metrics-ttm/{symbol.upper()}")
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return None
