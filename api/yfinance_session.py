"""Re-export for API layer. Session is defined in core so DataFetcher can use it without circular imports."""
from core.yfinance_session import get_yfinance_session

__all__ = ["get_yfinance_session"]
