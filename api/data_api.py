"""
Data API

Endpoints for unified data access (macro, sample data source).
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
router = APIRouter()


def _series_to_list(series) -> List[Dict[str, Any]]:
    """Convert pandas Series to list of {date, value} for JSON."""
    if series is None or series.empty:
        return []
    return [
        {"date": str(idx)[:10], "value": float(val)}
        for idx, val in series.items()
        if val is not None and str(val) != "nan"
    ]


@router.get("/macro")
async def get_macro() -> Dict[str, Any]:
    """
    Get macroeconomic indicators (FRED).
    Requires FRED_API_KEY. Returns series with latest values for dashboard.
    """
    try:
        from core.data_fetcher import DataFetcher
        try:
            from config import get_settings
            if not get_settings().data.fred_configured:
                return {"error": "FRED API key not configured. Set FRED_API_KEY in .env", "series": []}
        except ImportError:
            pass
        fetcher = DataFetcher()
        if not fetcher.fred:
            return {
                "error": "FRED API key not configured. Set FRED_API_KEY in .env",
                "series": [],
            }

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        series_list = []
        labels = [
            ("unemployment", "UNRATE", "Unemployment Rate"),
            ("cpi", "CPIAUCSL", "CPI"),
            ("fed_funds", "FEDFUNDS", "Fed Funds Rate"),
            ("treasury_10y", "DGS10", "10Y Treasury"),
        ]
        for key, sid, desc in labels:
            try:
                s = fetcher.get_economic_indicator(sid, start_date, end_date)
                data = _series_to_list(s)
                if data:
                    series_list.append(
                        {"series_id": sid, "description": desc, "data": data[-60:]}
                    )
            except Exception as e:
                logger.warning(f"Macro series {sid} failed: {e}")

        return {"series": series_list}
    except Exception as e:
        logger.warning(f"Macro endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
