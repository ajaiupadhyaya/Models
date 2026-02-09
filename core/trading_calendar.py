"""
Trading calendar awareness - avoid backtesting on non-trading days.
Uses exchange-calendars for NYSE, NASDAQ, LSE, etc.
"""

import pandas as pd
from exchange_calendars import get_calendar
from typing import List, Optional


class TradingCalendar:
    """Multi-exchange trading calendar."""
    
    SUPPORTED_EXCHANGES = {
        'NYSE': 'New York Stock Exchange',
        'NASDAQ': 'NASDAQ',
        'LSE': 'London Stock Exchange',
        'TSE': 'Tokyo Stock Exchange',
        'HK': 'Hong Kong Stock Exchange'
    }
    
    def __init__(self, exchange: str = 'NYSE'):
        """
        Initialize trading calendar.
        
        Args:
            exchange: Exchange code (NYSE, NASDAQ, etc.)
        """
        if exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Unknown exchange: {exchange}. Supported: {list(self.SUPPORTED_EXCHANGES.keys())}")
        
        self.exchange = exchange
        self.calendar = get_calendar(exchange)
    
    def trading_days(self, start: str, end: str) -> pd.DatetimeIndex:
        """
        Get all trading days in range.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            DatetimeIndex of trading days
        """
        return self.calendar.sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end)
        )
    
    def is_trading_day(self, date: str) -> bool:
        """Check if date is a trading day."""
        ts = pd.Timestamp(date)
        sessions = self.calendar.sessions
        if ts.tzinfo is None:
            ts = ts.tz_localize(sessions.tz)
        else:
            ts = ts.tz_convert(sessions.tz)
        return ts in sessions
    
    def next_trading_day(self, date: str) -> str:
        """Get next trading day after date."""
        ts = pd.Timestamp(date)
        sessions = self.calendar.sessions
        if ts.tzinfo is None:
            ts = ts.tz_localize(sessions.tz)
        else:
            ts = ts.tz_convert(sessions.tz)
        
        next_sessions = sessions[sessions > ts]
        if len(next_sessions) == 0:
            return str(sessions[-1].date())
        return str(next_sessions[0].date())
    
    def previous_trading_day(self, date: str) -> str:
        """Get previous trading day before date."""
        ts = pd.Timestamp(date)
        sessions = self.calendar.sessions
        if ts.tzinfo is None:
            ts = ts.tz_localize(sessions.tz)
        else:
            ts = ts.tz_convert(sessions.tz)
        
        prev_sessions = sessions[sessions < ts]
        if len(prev_sessions) == 0:
            return str(sessions[0].date())
        return str(prev_sessions[-1].date())
    
    def trading_days_between(self, start: str, end: str, count: int = 1) -> List[str]:
        """Get N trading days in range."""
        days = self.trading_days(start, end)
        return [str(d.date()) for d in days[:count]]
    
    def business_days_count(self, start: str, end: str) -> int:
        """Count trading days between two dates."""
        days = self.trading_days(start, end)
        return len(days)
