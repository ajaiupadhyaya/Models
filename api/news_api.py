"""
News API

Endpoints for real financial news (Finnhub). Cached to respect rate limits.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
import requests

from api.cache import get_cached, set_cached, cache_key

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_TTL_NEWS = 600  # 10 min


def _get_finnhub_key() -> Optional[str]:
    try:
        from config import get_settings
        return getattr(get_settings(), "finnhub_api_key", None) or os.environ.get("FINNHUB_API_KEY")
    except ImportError:
        return os.environ.get("FINNHUB_API_KEY")


@router.get("/news")
async def get_news(
    symbol: str = Query("AAPL", description="Stock symbol"),
    limit: int = Query(15, ge=1, le=50, description="Max number of articles"),
    days_back: int = Query(7, ge=1, le=30, description="Days to look back"),
) -> Dict[str, Any]:
    """
    Get company news from Finnhub.
    Requires FINNHUB_API_KEY. Returns title, summary, url, published. Cached 10 min.
    """
    key = cache_key("news", symbol, str(limit), str(days_back))
    cached = get_cached(key)
    if cached is not None:
        return cached

    api_key = _get_finnhub_key()
    if not api_key or not api_key.strip():
        return {
            "items": [],
            "error": "FINNHUB_API_KEY not configured. Set it in .env for real news. See .env.example.",
        }

    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date,
            "token": api_key,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            return {"items": [], "error": "Unexpected response from news API"}

        items: List[Dict[str, Any]] = []
        for art in raw[:limit]:
            headline = art.get("headline") or art.get("summary") or ""
            summary = art.get("summary") or headline
            url_link = art.get("url") or ""
            published = art.get("datetime")
            if published:
                try:
                    ts = int(published)
                    published = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, ValueError):
                    published = str(published)
            else:
                published = ""
            items.append({
                "title": headline,
                "summary": summary,
                "url": url_link,
                "published": published,
                "source": art.get("source", ""),
            })

        result = {"items": items, "symbol": symbol.upper()}
        set_cached(key, result, CACHE_TTL_NEWS)
        return result
    except requests.RequestException as e:
        logger.warning("News fetch failed for %s: %s", symbol, e)
        return {"items": [], "error": str(e), "symbol": symbol.upper()}
    except Exception as e:
        logger.warning("News processing failed: %s", e)
        return {"items": [], "error": str(e), "symbol": symbol.upper()}
