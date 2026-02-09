"""
NewsAPI data provider.

Supports: News articles with sentiment analysis
Docs: https://newsapi.org/docs
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import requests

from .base import DataProvider, OHLCV, FundamentalsData, AssetType

logger = logging.getLogger(__name__)


class NewsAPIProvider(DataProvider):
    """NewsAPI.org news provider."""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("newsapi", api_key or os.getenv("NEWSAPI_KEY"))
        self.rate_limit = 500  # Daily limit on free tier
        if not self.api_key:
            logger.warning("NEWSAPI_KEY not found; NewsAPI provider disabled")
    
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """NewsAPI doesn't provide OHLCV; news only."""
        return False  # Only fetch news via fetch_news()
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> List[OHLCV]:
        """NewsAPI doesn't support OHLCV."""
        raise NotImplementedError("NewsAPI provides news only; use fetch_news()")
    
    def fetch_latest_price(self, symbol: str) -> float:
        """NewsAPI doesn't support prices."""
        raise NotImplementedError("NewsAPI provides news only")
    
    def fetch_news(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch news articles about a symbol."""
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        # NewsAPI endpoints:
        # /everything - search articles (free tier has older data)
        # /top-headlines - recent top stories
        
        url = f"{self.BASE_URL}/everything"
        
        # Search query: include symbol and related terms
        query = f"{symbol} stock OR {symbol} equity"
        
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
            "apiKey": self.api_key,
        }
        
        # Add date filter if reasonable window
        days_back = 90  # Default to last 90 days
        from_date = datetime.now() - timedelta(days=days_back)
        params["from"] = from_date.isoformat()
        
        articles = []
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            
            if resp.status_code == 401:
                logger.error("Invalid NewsAPI key")
                return []
            
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") != "ok":
                logger.warning(f"NewsAPI error: {data.get('message')}")
                return []
            
            # Parse articles
            for item in data.get("articles", []):
                article = {
                    "title": item.get("title"),
                    "source": item.get("source", {}).get("name"),
                    "date": item.get("publishedAt"),
                    "url": item.get("url"),
                    "summary": item.get("description", ""),
                    "image": item.get("urlToImage"),
                    "content": item.get("content", ""),
                }
                articles.append(article)
            
            logger.info(f"NewsAPI: fetched {len(articles)} articles for {symbol}")
            return articles
        
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []
    
    def validate_api_key(self) -> bool:
        """Validate API key."""
        if not self.api_key:
            return False
        
        try:
            url = f"{self.BASE_URL}/top-headlines"
            resp = requests.get(
                url,
                params={"country": "us", "apiKey": self.api_key},
                timeout=5
            )
            return resp.status_code == 200
        except:
            return False
