"""
Tests for UnifiedDataFetcher V2.

Run with: pytest tests/test_unified_fetcher.py -v
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from core.unified_fetcher import (
    UnifiedDataFetcher,
    FetchRequest,
    FetchResult,
    ProviderMetrics,
    ProviderSelector,
)
from core.data_providers import (
    DataProviderRegistry,
    DataProvider,
    OHLCV,
    FundamentalsData,
    AssetType,
)


class TestFetchRequest:
    """Test FetchRequest dataclass."""
    
    def test_request_creation(self):
        """Create fetch request."""
        req = FetchRequest(
            symbol="AAPL",
            asset_type=AssetType.EQUITY,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
        )
        
        assert req.symbol == "AAPL"
        assert req.asset_type == AssetType.EQUITY
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        req = FetchRequest(
            symbol="AAPL",
            asset_type=AssetType.EQUITY,
        )
        
        key = req.to_cache_key()
        assert "AAPL" in key
        assert "equity" in key


class TestFetchResult:
    """Test FetchResult dataclass."""
    
    def test_result_to_dict(self):
        """Convert result to dict."""
        result = FetchResult(
            symbol="AAPL",
            asset_type=AssetType.EQUITY,
            provider="polygon",
            cache_hit=False,
            latency_ms=250.0,
            cost_cents=2.0,
        )
        
        d = result.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["provider"] == "polygon"
        assert d["cache_hit"] is False


class TestProviderMetrics:
    """Test ProviderMetrics tracking."""
    
    def test_metrics_creation(self):
        """Create provider metrics."""
        metrics = ProviderMetrics(name="polygon")
        
        assert metrics.name == "polygon"
        assert metrics.success_rate == 0.0
        assert metrics.avg_latency_ms == 0.0
    
    def test_success_rate_calculation(self):
        """Calculate success rate."""
        metrics = ProviderMetrics(name="polygon")
        metrics.total_requests = 10
        metrics.successful_requests = 8
        
        assert metrics.success_rate == 0.8
    
    def test_avg_latency(self):
        """Calculate average latency."""
        metrics = ProviderMetrics(name="polygon")
        metrics.successful_requests = 5
        metrics.total_latency_ms = 1000.0  # 5 requests × 200ms each
        
        assert metrics.avg_latency_ms == 200.0


class TestProviderSelector:
    """Test intelligent provider selection."""
    
    def test_selector_creation(self):
        """Create provider selector."""
        registry = DataProviderRegistry()
        metrics = {}
        selector = ProviderSelector(registry, metrics)
        
        assert selector.registry == registry
    
    def test_asset_type_preferences(self):
        """Test provider preferences for asset types."""
        selector = ProviderSelector(DataProviderRegistry(), {})
        
        equity_prefs = selector.PREFERENCES[AssetType.EQUITY]
        assert "polygon" in equity_prefs
        assert "iex" in equity_prefs
        
        crypto_prefs = selector.PREFERENCES[AssetType.CRYPTO]
        assert "coingecko" in crypto_prefs
    
    def test_provider_scoring(self):
        """Test provider health scoring."""
        registry = DataProviderRegistry()
        
        # Create mock providers
        mock_provider = Mock(spec=DataProvider)
        mock_provider.name = "polygon"
        mock_provider.supports_asset_type.return_value = True
        registry.register(mock_provider)
        
        # Create metrics: high success rate
        metrics = {
            "polygon": ProviderMetrics(
                name="polygon",
                total_requests=100,
                successful_requests=95,
                total_latency_ms=20000,  # avg 200ms
            ),
            "iex": ProviderMetrics(
                name="iex",
                total_requests=100,
                successful_requests=90,
                total_latency_ms=10000,  # avg 100ms
            ),
        }
        
        selector = ProviderSelector(registry, metrics)
        # Provider with higher success rate should score better
        polygon_score = metrics["polygon"].success_rate
        iex_score = metrics["iex"].success_rate
        assert polygon_score > iex_score


class TestUnifiedDataFetcher:
    """Test UnifiedDataFetcher V2."""
    
    def test_fetcher_creation(self):
        """Create unified fetcher."""
        fetcher = UnifiedDataFetcher()
        
        assert fetcher.registry is not None
        assert fetcher.selector is not None
        assert len(fetcher.metrics) > 0
    
    def test_custom_registry(self):
        """Create with custom registry."""
        registry = DataProviderRegistry()
        fetcher = UnifiedDataFetcher(registry=registry)
        
        assert fetcher.registry == registry
    
    @patch("core.unified_fetcher.DataProviderRegistry")
    def test_fetch_ohlcv_success(self, mock_registry_class):
        """Test successful OHLCV fetch."""
        # Create mock provider
        mock_provider = Mock(spec=DataProvider)
        mock_provider.name = "polygon"
        mock_provider.supports_asset_type.return_value = True
        mock_provider.fetch_ohlcv.return_value = [
            OHLCV(
                date=datetime(2025, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=1000000,
            )
        ]
        
        # Create mock registry
        mock_registry = Mock(spec=DataProviderRegistry)
        mock_registry.get_provider.return_value = mock_provider
        
        fetcher = UnifiedDataFetcher(registry=mock_registry)
        
        result = fetcher.fetch_ohlcv(
            "AAPL",
            AssetType.EQUITY,
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
        )
        
        assert result.symbol == "AAPL"
        assert result.asset_type == AssetType.EQUITY
        assert len(result.data) == 1
        assert result.error is None
    
    def test_fetch_ohlcv_all_providers_fail(self):
        """Test fallback when all providers fail."""
        # Create mock registry that returns None (no providers)
        mock_registry = Mock(spec=DataProviderRegistry)
        mock_registry.get_provider.return_value = None
        
        fetcher = UnifiedDataFetcher(registry=mock_registry)
        
        with pytest.raises(ConnectionError):
            fetcher.fetch_ohlcv(
                "AAPL",
                AssetType.EQUITY,
            )
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        fetcher = UnifiedDataFetcher()
        
        # First call should succeed
        result1 = fetcher._check_rate_limit("coingecko")
        assert result1 is True
        
        # Immediate second call should also succeed (with sleep)
        result2 = fetcher._check_rate_limit("coingecko")
        assert result2 is True
    
    def test_audit_log_recording(self):
        """Test audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = UnifiedDataFetcher()
            fetcher.audit_log_path = Path(tmpdir) / "audit.jsonl"
            
            result = FetchResult(
                symbol="AAPL",
                asset_type=AssetType.EQUITY,
                provider="polygon",
                latency_ms=250.0,
            )
            
            fetcher._record_fetch(result)
            
            # Check file exists and contains entry
            assert fetcher.audit_log_path.exists()
            
            with open(fetcher.audit_log_path) as f:
                line = f.readline()
                entry = json.loads(line)
                assert entry["symbol"] == "AAPL"
                assert entry["provider"] == "polygon"
    
    def test_metrics_update(self):
        """Test metrics tracking."""
        fetcher = UnifiedDataFetcher()
        
        result = FetchResult(
            symbol="AAPL",
            asset_type=AssetType.EQUITY,
            provider="polygon",
            latency_ms=250.0,
            cost_cents=2.0,
        )
        
        initial_total = fetcher.metrics["polygon"].total_requests
        fetcher._record_fetch(result)
        
        assert fetcher.metrics["polygon"].total_requests == initial_total + 1
        assert fetcher.metrics["polygon"].successful_requests == 1
    
    def test_get_metrics(self):
        """Get provider metrics."""
        fetcher = UnifiedDataFetcher()
        
        # Record a fetch
        result = FetchResult(
            symbol="AAPL",
            asset_type=AssetType.EQUITY,
            provider="polygon",
            latency_ms=250.0,
        )
        fetcher._record_fetch(result)
        
        metrics = fetcher.get_metrics()
        
        assert "polygon" in metrics
        assert "success_rate" in metrics["polygon"]
        assert "avg_latency_ms" in metrics["polygon"]
    
    def test_audit_log_retrieval(self):
        """Retrieve audit log entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = UnifiedDataFetcher()
            fetcher.audit_log_path = Path(tmpdir) / "audit.jsonl"
            
            # Write some entries
            for i in range(5):
                result = FetchResult(
                    symbol=f"TICK{i}",
                    asset_type=AssetType.EQUITY,
                    provider="polygon",
                )
                fetcher._record_fetch(result)
            
            # Retrieve last 3
            entries = fetcher.get_audit_log(limit=3)
            
            assert len(entries) <= 3
            assert all("symbol" in e for e in entries)
    
    @patch("core.unified_fetcher.DataProviderRegistry")
    def test_fetch_latest_price(self, mock_registry_class):
        """Test price fetching."""
        mock_provider = Mock(spec=DataProvider)
        mock_provider.name = "polygon"
        mock_provider.supports_asset_type = Mock(return_value=True)
        mock_provider.fetch_latest_price = Mock(return_value=150.0)
        
        mock_registry = Mock(spec=DataProviderRegistry)
        # Make get_provider return the mock provider for providers in EQUITY preferences
        def get_provider_side_effect(name):
            if name in ["polygon", "iex", "yfinance"]:
                return mock_provider
            return None
        mock_registry.get_provider = Mock(side_effect=get_provider_side_effect)
        
        fetcher = UnifiedDataFetcher(registry=mock_registry)
        price, result = fetcher.fetch_latest_price("AAPL", AssetType.EQUITY)
        
        assert price == 150.0
        assert result.provider in ["polygon", "iex", "yfinance"]
    
    @patch("core.unified_fetcher.DataProviderRegistry")
    def test_fetch_fundamentals(self, mock_registry_class):
        """Test fundamentals fetching."""
        fundamentals = FundamentalsData(
            symbol="AAPL",
            price=150.0,
            pe_ratio=25.5,
            market_cap=2500000000000,
        )
        
        mock_provider = Mock(spec=DataProvider)
        mock_provider.name = "iex"
        mock_provider.supports_asset_type = Mock(return_value=True)
        mock_provider.fetch_fundamentals = Mock(return_value=fundamentals)
        
        mock_registry = Mock(spec=DataProviderRegistry)
        # Make get_provider return the mock provider for providers in EQUITY preferences  
        def get_provider_side_effect(name):
            if name in ["polygon", "iex", "yfinance"]:
                return mock_provider
            return None
        mock_registry.get_provider = Mock(side_effect=get_provider_side_effect)
        
        fetcher = UnifiedDataFetcher(registry=mock_registry)
        fund, result = fetcher.fetch_fundamentals("AAPL", AssetType.EQUITY)
        
        assert fund is not None
        assert fund.symbol == "AAPL"
        assert fund.pe_ratio == 25.5
        assert result.provider in ["polygon", "iex", "yfinance"]


class TestIntegration:
    """Integration tests with real (or mocked) API calls."""
    
    def test_multi_provider_fallback(self):
        """Test that fetcher uses fallback on first provider failure."""
        # Create fallback providers
        primary = Mock(spec=DataProvider)
        primary.name = "polygon"
        primary.supports_asset_type.return_value = True
        primary.fetch_ohlcv.side_effect = ConnectionError("API down")
        
        secondary = Mock(spec=DataProvider)
        secondary.name = "iex"
        secondary.supports_asset_type.return_value = True
        secondary.fetch_ohlcv.return_value = [
            OHLCV(
                date=datetime(2025, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=1000000,
            )
        ]
        
        registry = DataProviderRegistry()
        registry.register(primary)
        registry.register(secondary)
        registry.set_fallback_chain(["polygon", "iex"])
        
        fetcher = UnifiedDataFetcher(registry=registry)
        # Manually set selector to use these providers
        fetcher.selector.PREFERENCES[AssetType.EQUITY] = ["polygon", "iex"]
        
        result = fetcher.fetch_ohlcv(
            "AAPL",
            AssetType.EQUITY,
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
        )
        
        # Should have used secondary (iex) due to primary failure
        assert result.provider == "iex"
        assert len(result.data) == 1
    
    def test_cost_tracking(self):
        """Test cost estimation and tracking."""
        mock_provider = Mock(spec=DataProvider)
        mock_provider.name = "polygon"
        mock_provider.supports_asset_type.return_value = True
        mock_provider.fetch_ohlcv.return_value = [
            OHLCV(
                date=datetime(2025, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=1000000,
            )
        ]
        
        registry = DataProviderRegistry()
        registry.register(mock_provider)
        
        fetcher = UnifiedDataFetcher(registry=registry)
        result = fetcher.fetch_ohlcv(
            "AAPL",
            AssetType.EQUITY,
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
        )
        
        # Polygon costs 2 cents per 1000 requests
        assert result.cost_cents > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
