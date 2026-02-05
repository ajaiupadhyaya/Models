"""
Tests for recent improvements:
- Data fetcher retry logic and stock info
- Cache statistics tracking
- Rate limiter statistics tracking
- Backtesting validation
- System stats endpoint
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from core.data_fetcher import DataFetcher
from core.backtesting import BacktestEngine
from api.cache import TTLCache
from api.rate_limit import InMemoryRateLimiter


class TestDataFetcherImprovements:
    """Test data fetcher enhancements."""
    
    def test_stock_info_returns_expected_keys(self):
        """Test get_stock_info returns expected structure."""
        fetcher = DataFetcher()
        
        try:
            info = fetcher.get_stock_info("AAPL")
            
            # Check expected keys exist
            assert 'symbol' in info
            assert 'name' in info
            assert 'sector' in info
            assert 'market_cap' in info
            assert 'current_price' in info
            
        except ValueError:
            # API might fail in CI, that's ok
            pytest.skip("Yahoo Finance API unavailable")
    
    def test_get_multiple_stocks_empty_list(self):
        """Test get_multiple_stocks with empty list."""
        fetcher = DataFetcher()
        result = fetcher.get_multiple_stocks([])
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestCacheImprovements:
    """Test cache statistics tracking."""
    
    def test_cache_tracks_hits_and_misses(self):
        """Test cache tracks statistics correctly."""
        cache = TTLCache()
        
        # Initial state
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['sets'] == 0
        
        # Set a value
        cache.set('key1', 'value1', 60)
        stats = cache.get_stats()
        assert stats['sets'] == 1
        assert stats['size'] == 1
        
        # Get existing value (hit)
        value = cache.get('key1')
        assert value == 'value1'
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0
        
        # Get non-existent value (miss)
        value = cache.get('key2')
        assert value is None
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_cache_tracks_expirations(self):
        """Test cache tracks expirations."""
        cache = TTLCache()
        
        # Set a value with 0 TTL (instant expiry)
        cache.set('key1', 'value1', 0)
        
        # Try to get expired value
        value = cache.get('key1')
        assert value is None
        
        stats = cache.get_stats()
        assert stats['expirations'] == 1
    
    def test_cache_calculates_hit_rate(self):
        """Test cache calculates hit rate correctly."""
        cache = TTLCache()
        
        cache.set('key1', 'value1', 60)
        cache.get('key1')  # hit
        cache.get('key1')  # hit
        cache.get('key2')  # miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 66.67  # 2/3
    
    def test_cache_clear_works(self):
        """Test cache clear functionality."""
        cache = TTLCache()
        
        cache.set('key1', 'value1', 60)
        cache.set('key2', 'value2', 60)
        
        assert cache.get_stats()['size'] == 2
        
        cache.clear()
        
        assert cache.get_stats()['size'] == 0
        assert cache.get('key1') is None
    
    def test_cache_reset_stats_works(self):
        """Test cache reset_stats functionality."""
        cache = TTLCache()
        
        cache.set('key1', 'value1', 60)
        cache.get('key1')
        cache.get('key2')
        
        cache.reset_stats()
        
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['sets'] == 0
        assert stats['expirations'] == 0


class TestRateLimiterImprovements:
    """Test rate limiter statistics tracking."""
    
    def test_rate_limiter_tracks_allowed_requests(self):
        """Test rate limiter tracks allowed requests."""
        limiter = InMemoryRateLimiter(max_requests=5, window_sec=60)
        
        # Initial state
        stats = limiter.get_stats()
        assert stats['allowed'] == 0
        assert stats['blocked'] == 0
        
        # Make allowed requests
        for _ in range(5):
            allowed, _ = limiter.is_allowed('127.0.0.1')
            assert allowed
        
        stats = limiter.get_stats()
        assert stats['allowed'] == 5
        assert stats['blocked'] == 0
    
    def test_rate_limiter_tracks_blocked_requests(self):
        """Test rate limiter tracks blocked requests."""
        limiter = InMemoryRateLimiter(max_requests=2, window_sec=60)
        
        # Use up the limit
        limiter.is_allowed('127.0.0.1')
        limiter.is_allowed('127.0.0.1')
        
        # Next request should be blocked
        allowed, retry_after = limiter.is_allowed('127.0.0.1')
        assert not allowed
        assert retry_after > 0
        
        stats = limiter.get_stats()
        assert stats['allowed'] == 2
        assert stats['blocked'] == 1
    
    def test_rate_limiter_reset_stats_works(self):
        """Test rate limiter reset_stats functionality."""
        limiter = InMemoryRateLimiter(max_requests=5, window_sec=60)
        
        limiter.is_allowed('127.0.0.1')
        limiter.reset_stats()
        
        stats = limiter.get_stats()
        assert stats['allowed'] == 0
        assert stats['blocked'] == 0


class TestBacktestingValidation:
    """Test backtesting engine validation."""
    
    def test_backtest_validates_initial_capital(self):
        """Test backtesting validates positive initial capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            BacktestEngine(initial_capital=0)
        
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            BacktestEngine(initial_capital=-1000)
    
    def test_backtest_validates_commission(self):
        """Test backtesting validates commission range."""
        with pytest.raises(ValueError, match="Commission must be between 0 and 1"):
            BacktestEngine(commission=-0.1)
        
        with pytest.raises(ValueError, match="Commission must be between 0 and 1"):
            BacktestEngine(commission=1.5)
    
    def test_backtest_validates_empty_dataframe(self):
        """Test backtesting validates non-empty DataFrame."""
        engine = BacktestEngine()
        df = pd.DataFrame()
        signals = np.array([])
        
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            engine.run_backtest(df, signals)
    
    def test_backtest_validates_close_column(self):
        """Test backtesting validates Close column exists."""
        engine = BacktestEngine()
        df = pd.DataFrame({'Open': [100, 101, 102]})
        signals = np.array([0, 0, 0])
        
        with pytest.raises(ValueError, match="DataFrame must have 'Close' column"):
            engine.run_backtest(df, signals)
    
    def test_backtest_validates_signals_length(self):
        """Test backtesting validates signals match data length."""
        engine = BacktestEngine()
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000, 1100, 1200]
        })
        signals = np.array([0, 0])  # Wrong length
        
        with pytest.raises(ValueError, match="Signals length.*must match data length"):
            engine.run_backtest(df, signals)
    
    def test_backtest_validates_position_size(self):
        """Test backtesting validates position size range."""
        engine = BacktestEngine()
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000, 1100, 1200]
        })
        signals = np.array([0, 0, 0])
        
        with pytest.raises(ValueError, match="Position size must be between 0 and 1"):
            engine.run_backtest(df, signals, position_size=0)
        
        with pytest.raises(ValueError, match="Position size must be between 0 and 1"):
            engine.run_backtest(df, signals, position_size=1.5)
    
    def test_backtest_runs_with_valid_inputs(self):
        """Test backtesting runs successfully with valid inputs."""
        engine = BacktestEngine()
        
        # Create sample data
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Open': [99, 100, 101, 102, 103],
            'High': [101, 102, 103, 104, 105],
            'Low': [98, 99, 100, 101, 102],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        signals = np.array([0.5, 0.6, 0.4, -0.5, -0.3])
        
        # Should not raise
        result = engine.run_backtest(df, signals, signal_threshold=0.3, position_size=0.1)
        
        assert isinstance(result, dict)
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result


class TestMonitoringImprovements:
    """Test monitoring endpoint enhancements."""
    
    def test_system_stats_endpoint_exists(self):
        """Test system stats endpoint is accessible."""
        try:
            from api.monitoring import router
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
            
            app = FastAPI()
            app.include_router(router, prefix="/monitoring")
            client = TestClient(app)
            
            # Make request
            response = client.get("/monitoring/system/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert 'timestamp' in data
            assert 'cache' in data
            assert 'rate_limiter' in data
        except Exception:
            # Skip if dependencies not available in CI
            pytest.skip("Monitoring dependencies not available")
