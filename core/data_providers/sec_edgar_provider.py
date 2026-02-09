"""
SEC EDGAR data provider.

Supports: US company fundamentals (10-K, 10-Q filings)
Docs: https://www.sec.gov/cgi-bin/browse-edgar
Free tier: No API key required
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import requests
import json

from .base import DataProvider, OHLCV, FundamentalsData, AssetType

logger = logging.getLogger(__name__)


class SECEdgarProvider(DataProvider):
    """SEC EDGAR fundamentals provider."""
    
    BASE_URL = "https://data.sec.gov/api/xbrl"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    
    def __init__(self):
        super().__init__("sec_edgar")
        self.rate_limit = 10  # Requests per second
        self.timeout = 30
        # CIK lookup cache
        self.cik_cache: Dict[str, str] = {}
    
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """SEC EDGAR supports US equities fundamentals."""
        return asset_type == AssetType.EQUITY
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> List[OHLCV]:
        """SEC EDGAR doesn't provide OHLCV."""
        raise NotImplementedError("SEC EDGAR provides fundamentals only")
    
    def fetch_latest_price(self, symbol: str) -> float:
        """SEC EDGAR doesn't provide prices."""
        raise NotImplementedError("SEC EDGAR provides fundamentals only")
    
    def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a stock symbol."""
        # Check cache
        if symbol in self.cik_cache:
            return self.cik_cache[symbol]
        
        try:
            # SEC ticker lookup
            url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": symbol,
                "type": "",
                "dateb": "",
                "owner": "exclude",
                "count": 1,
                "output": "json",
            }
            
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            if "CIK_list" in data and data["CIK_list"]:
                cik = str(data["CIK_list"][0]["CIK"])
                self.cik_cache[symbol] = cik
                return cik
            
            logger.warning(f"Could not find CIK for {symbol}")
            return None
        
        except Exception as e:
            logger.warning(f"CIK lookup error for {symbol}: {e}")
            return None
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalsData]:
        """Fetch latest fundamentals from SEC EDGAR."""
        try:
            # Get CIK
            cik = self._get_cik(symbol)
            if not cik:
                return None
            
            # Pad CIK to 10 digits
            cik = cik.zfill(10)
            
            # Fetch company facts (aggregated financials)
            # https://data.sec.gov/api/xbrl/companyfacts/CIK0000000051.json
            url = f"{self.BASE_URL}/companyfacts/CIK{cik}.json"
            resp = requests.get(url, timeout=self.timeout)
            
            if resp.status_code == 404:
                logger.warning(f"Company facts not found for CIK {cik}")
                return None
            
            resp.raise_for_status()
            data = resp.json()
            
            # Extract latest financials from US-GAAP taxonomy
            us_gaap = data.get("facts", {}).get("us-gaap", {})
            
            # Get latest values from various metrics
            def get_latest_value(key: str) -> Optional[float]:
                """Extract latest non-zero value."""
                if key not in us_gaap:
                    return None
                
                facts = us_gaap[key].get("units", {}).get("USD", [])
                if not facts:
                    return None
                
                # Get most recent filing
                for fact in reversed(sorted(facts, key=lambda x: x.get("filed", ""))):
                    val = fact.get("val")
                    if val and val != 0:
                        return float(val)
                
                return None
            
            # Extract key metrics
            market_cap = get_latest_value("EntityMarketCapitalization")
            net_income = get_latest_value("NetIncomeLoss")
            revenue = get_latest_value("Revenues")
            assets = get_latest_value("Assets")
            liabilities = get_latest_value("Liabilities")
            stockholders_equity = get_latest_value("StockholdersEquity")
            shares_outstanding = get_latest_value("EntityCommonStockSharesOutstanding")
            
            # Calculate derived metrics if possible
            eps = None
            if net_income and shares_outstanding and shares_outstanding > 0:
                eps = net_income / shares_outstanding
            
            book_value_per_share = None
            if stockholders_equity and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = stockholders_equity / shares_outstanding
            
            # Note: We can't get current price from SEC EDGAR
            return FundamentalsData(
                symbol=symbol,
                price=0.0,  # SEC doesn't provide price
                market_cap=market_cap,
                net_income=net_income,
                revenue=revenue,
                earnings_per_share=eps,
                book_value_per_share=book_value_per_share,
            )
        
        except Exception as e:
            logger.warning(f"SEC EDGAR fundamentals error: {e}")
            return None
    
    def validate_api_key(self) -> bool:
        """SEC EDGAR doesn't require API key."""
        try:
            # Test with known company (Apple Inc.)
            url = "https://www.sec.gov/cgi-bin/browse-edgar"
            resp = requests.get(
                url,
                params={
                    "action": "getcompany",
                    "CIK": "AAPL",
                    "count": 1,
                    "output": "json",
                },
                timeout=5,
            )
            return resp.status_code == 200
        except:
            return False
