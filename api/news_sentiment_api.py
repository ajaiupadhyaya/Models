"""
News & Sentiment API: per-ticker news with VADER sentiment scoring.
GET /api/v1/news/{symbol} - news feed + 7-day aggregate sentiment gauge.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_TTL = 600


def _get_finnhub_key() -> Optional[str]:
    try:
        from config import get_settings
        return getattr(get_settings(), "finnhub_api_key", None) or os.environ.get("FINNHUB_API_KEY")
    except ImportError:
        return os.environ.get("FINNHUB_API_KEY")


def _get_newsapi_key() -> Optional[str]:
    return os.environ.get("NEWSAPI_KEY")


def _score_sentiment_vader(text: str) -> float:
    """Score text -1 (negative) to 1 (positive) using VADER."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text or "")
        compound = scores.get("compound", 0)
        return float(compound)
    except ImportError:
        return 0.0


def _fetch_news_finnhub(symbol: str, days: int) -> List[Dict[str, Any]]:
    """Fetch news from Finnhub."""
    key = _get_finnhub_key()
    if not key or not key.strip():
        return []
    try:
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": symbol.upper(), "from": from_date, "to": to_date, "token": key}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            return []
        items = []
        for art in raw[:30]:
            headline = art.get("headline") or art.get("summary") or ""
            summary = art.get("summary") or headline
            pub = art.get("datetime")
            if pub:
                try:
                    pub = datetime.utcfromtimestamp(int(pub)).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, ValueError):
                    pub = str(pub)
            else:
                pub = ""
            items.append({
                "title": headline,
                "summary": summary,
                "url": art.get("url") or "",
                "published": pub,
                "source": art.get("source", ""),
            })
        return items
    except Exception as e:
        logger.warning("Finnhub news fetch failed: %s", e)
        return []


def _fetch_news_newsapi(symbol: str, days: int) -> List[Dict[str, Any]]:
    """Fetch news from NewsAPI."""
    key = _get_newsapi_key()
    if not key or not key.strip():
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        params = {
            "q": symbol,
            "from": from_date,
            "sortBy": "publishedAt",
            "apiKey": key,
            "language": "en",
            "pageSize": 30,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles") or []
        items = []
        for a in articles:
            items.append({
                "title": (a.get("title") or "")[:500],
                "summary": (a.get("description") or a.get("title") or "")[:1000],
                "url": a.get("url") or "",
                "published": a.get("publishedAt") or "",
                "source": (a.get("source") or {}).get("name", ""),
            })
        return items
    except Exception as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return []


@router.get("/{symbol}")
async def get_news_sentiment(
    symbol: str,
    limit: int = 20,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Per-ticker news feed with sentiment score per article (VADER).
    Aggregate 7-day sentiment displayed as gauge value (-1 to 1).
    """
    sym = symbol.upper()
    items = _fetch_news_finnhub(sym, days)
    if not items:
        items = _fetch_news_newsapi(sym, days)

    if not items:
        return {
            "symbol": sym,
            "items": [],
            "aggregate_sentiment_7d": 0.0,
            "article_count": 0,
            "error": "No news sources configured (FINNHUB_API_KEY or NEWSAPI_KEY) or no articles found",
        }

    # Score each article
    scored = []
    for art in items[:limit]:
        text = (art.get("title") or "") + " " + (art.get("summary") or "")
        score = _score_sentiment_vader(text)
        scored.append({
            **art,
            "sentiment_score": round(score, 4),
        })

    # Aggregate sentiment (weighted by recency could be added; simple mean for now)
    scores = [a["sentiment_score"] for a in scored]
    agg = sum(scores) / len(scores) if scores else 0.0

    return {
        "symbol": sym,
        "items": scored,
        "aggregate_sentiment_7d": round(agg, 4),
        "article_count": len(scored),
    }
