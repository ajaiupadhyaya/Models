"""
Unit tests for data providers.

Tests can be run with API keys in environment:
  export POLYGON_API_KEY=pk_...
  export IEX_API_KEY=pk_...
  export NEWSAPI_KEY=sk_...
  pytest tests/test_data_providers.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import os

from core.data_providers.base import (
    DataProvider, OHLCV, FundamentalsData, AssetType, 
    DataProviderRegistry
)
from core.data_providers.polygon_provider import PolygonProvider
from core.data_providers.iex_provider import IEXProvider
from core.data_providers.coingecko_provider import CoinGeckoProvider
from core.data_providers.newsapi_provider import NewsAPIProvider
from core.data_providers.sec_edgar_provider import SECEdgarProvider


class TestOHLCV:
    """Test OHLCV dataclass."""
    
    def test_ohlcv_creation(self):
        """Test creating OHLCV bar."""
        date = datetime(2025, 1, 1, 10, 0, 0)
        ohlcv = OHLCV(
            date=date,
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000000,
        )
        
        assert ohlcv.date == date
        assert ohlcv.open == 100.0
        assert ohlcv.close == 102.0
        assert ohlcv.volume == 1000000
    
    def test_ohlcv_to_dict(self):
        """Test OHLCV serialization."""
        ohlcv = OHLCV(
            date=datetime(2025, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000000,
        )
        
        d = ohlcv.to_dict()
        assert d["open"] == 100.0
        assert d["close"] == 102.0
        assert "date" in d


class TestFundamentalsData:
    """Test FundamentalsData dataclass."""
    
    def test_fundamentals_creation(self):
        """Test creating fundamentals data."""
        fund = FundamentalsData(
            symbol="AAPL",
            price=150.0,
            pe_ratio=25.5,
            market_cap=2500000000000,
        )
        
        assert fund.symbol == "AAPL"
        assert fund.price == 150.0
        assert fund.pe_ratio == 25.5


class TestDataProviderRegistry:
    """Test provider registry and fallback."""
    
    def test_register_provider(self):
        """Test registering a provider."""
        registry = DataProviderRegistry()
        
        # Create mock provider
        mock_provider = Mock(spec=DataProvider)
        mock_provider.name = "test"
        
        registry.register(mock_provider)
        assert registry.get_provider("test") == mock_provider
    
    def test_fallback_chain(self):
        """Test fallback chain setting."""
        registry = DataProviderRegistry()
        registry.set_fallback_chain(["primary", "secondary", "tertiary"])
        
        assert registry.fallback_chain == ["primary", "secondary", "tertiary"]
    
    def test_fetch_with_fallback(self):
        """Test fetching with fallback to next provider."""
        registry = DataProviderRegistry()
        
        # Create mock providers
        primary = Mock(spec=DataProvider)
        primary.name = "primary"
        primary.supports_asset_type.return_value = True
        primary.fetch_ohlcv.side_effect = ConnectionError("API down")
        
        secondary = Mock(spec=DataProvider)
        secondary.name = "secondary"
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
        
        registry.register(primary)
        registry.register(secondary)
        registry.set_fallback_chain(["primary", "secondary"])
        
        # Fetch should try primary, fail, then use secondary
        ohlcv, provider_used = registry.fetch_ohlcv_with_fallback(
            "AAPL",
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
        )
        
        assert provider_used == "secondary"
        assert len(ohlcv) == 1


class TestPolygonProvider:
    """Test Polygon provider."""
    
    @patch("core.data_providers.polygon_provider.requests.get")
    def test_fetch_ohlcv_success(self, mock_get):
        """Test successful OHLCV fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "t": 1704067200000,  # 2024-01-01 in milliseconds
                    "o": 100.0,
                    "h": 105.0,
                    "l": 99.0,
                    "c": 102.0,
                    "v": 1000000,
                    "vw": 101.5,
                }
            ],
            "status": "OK",
        }
        mock_get.return_value = mock_response
        
        provider = PolygonProvider(api_key="test_key")
        ohlcv = provider.fetch_ohlcv(
            "AAPL",
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
        )
        
        assert len(ohlcv) == 1
        assert ohlcv[0].open == 100.0
        assert ohlcv[0].close == 102.0
    
    def test_supports_asset_types(self):
        """Test asset type support."""
        provider = PolygonProvider(api_key="test_key")
        
        assert provider.supports_asset_type(AssetType.EQUITY)
        assert provider.supports_asset_type(AssetType.CRYPTO)
        assert provider.supports_asset_type(AssetType.FOREX)
        assert not provider.supports_asset_type(AssetType.FIXED_INCOME)


class TestIEXProvider:
    """Test IEX provider."""
    
    @patch("core.data_providers.iex_provider.requests.get")
    def test_fetch_ohlcv_success(self, mock_get):
        """Test successful OHLCV fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "date": "2025-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1000000,
            }
        ]
        mock_get.return_value = mock_response
        
        provider = IEXProvider(api_key="test_key")
        ohlcv = provider.fetch_ohlcv(
            "AAPL",
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
        )
        
        assert len(ohlcv) == 1
        assert ohlcv[0].open == 100.0
    
    def test_only_supports_equities(self):
        """Test that IEX only supports equities."""
        provider = IEXProvider(api_key="test_key")
        
        assert provider.supports_asset_type(AssetType.EQUITY)
        assert not provider.supports_asset_type(AssetType.CRYPTO)
        assert not provider.supports_asset_type(AssetType.FOREX)


class TestCoinGeckoProvider:
    """Test CoinGecko crypto provider."""
    
    def test_supports_crypto_only(self):
        """Test CoinGecko only supports crypto."""
        provider = CoinGeckoProvider()
        
        assert provider.supports_asset_type(AssetType.CRYPTO)
        assert not provider.supports_asset_type(AssetType.EQUITY)
        assert not provider.supports_asset_type(AssetType.FOREX)
    
    def test_coin_id_mapping(self):
        """Test symbol to coin ID mapping."""
        provider = CoinGeckoProvider()
        
        assert provider._get_coin_id("BTC") == "bitcoin"
        assert provider._get_coin_id("ETH") == "ethereum"
        assert provider._get_coin_id("MATIC") == "matic-network"
    
    @patch("core.data_providers.coingecko_provider.requests.get")
    def test_fetch_latest_price(self, mock_get):
        """Test fetching latest price."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bitcoin": {"usd": 42000.0}
        }
        mock_get.return_value = mock_response
        
        provider = CoinGeckoProvider()
        price = provider.fetch_latest_price("BTC")
        
        assert price == 42000.0


class TestNewsAPIProvider:
    """Test NewsAPI provider."""
    
    def test_ohlcv_not_supported(self):
        """Test that NewsAPI doesn't support OHLCV."""
        provider = NewsAPIProvider(api_key="test_key")
        
        with pytest.raises(NotImplementedError):
            provider.fetch_ohlcv(
                "AAPL",
                datetime(2025, 1, 1),
                datetime(2025, 1, 31),
            )
    
    @patch("core.data_providers.newsapi_provider.requests.get")
    def test_fetch_news(self, mock_get):
        """Test fetching news articles."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Apple stock rises",
                    "source": {"name": "Financial Times"},
                    "publishedAt": "2025-01-01T10:00:00Z",
                    "url": "https://example.com/article",
                    "description": "Apple stock surges on new product",
                    "content": "Full article content...",
                }
            ],
        }
        mock_get.return_value = mock_response
        
        provider = NewsAPIProvider(api_key="test_key")
        articles = provider.fetch_news("AAPL", limit=5)
        
        assert len(articles) == 1
        assert articles[0]["title"] == "Apple stock rises"
        assert articles[0]["source"] == "Financial Times"


class TestSECEdgarProvider:
    """Test SEC EDGAR provider."""
    
    def test_ohlcv_not_supported(self):
        """Test that SEC doesn't support OHLCV."""
        provider = SECEdgarProvider()
        
        with pytest.raises(NotImplementedError):
            provider.fetch_ohlcv(
                "AAPL",
                datetime(2025, 1, 1),
                datetime(2025, 1, 31),
            )
    
    @patch("core.data_providers.sec_edgar_provider.requests.get")
    def test_get_cik(self, mock_get):
        """Test getting CIK for a symbol."""
        # First call returns CIK lookup
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "CIK_list": [{"CIK": 320193}]  # Apple's CIK
        }
        mock_get.return_value = mock_response
        
        provider = SECEdgarProvider()
        cik = provider._get_cik("AAPL")
        
        assert cik == "320193"


class TestProviderIntegration:
    """Integration tests (require API keys)."""
    
    @pytest.mark.skipif(
        not os.getenv("POLYGON_API_KEY"),
        reason="POLYGON_API_KEY not set"
    )
    def test_polygon_real_api(self):
        """Test Polygon with real API."""
        provider = PolygonProvider()
        
        # Get AAPL data for last 5 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        ohlcv = provider.fetch_ohlcv("AAPL", start_date, end_date, "1day")
        assert len(ohlcv) > 0
        assert all(isinstance(o, OHLCV) for o in ohlcv)
    
    @pytest.mark.skipif(
        not os.getenv("IEX_API_KEY"),
        reason="IEX_API_KEY not set"
    )
    def test_iex_real_api(self):
        """Test IEX with real API."""
        provider = IEXProvider()
        
        price = provider.fetch_latest_price("AAPL")
        assert price > 0
    
    def test_coingecko_real_api(self):
        """Test CoinGecko with real API (always available)."""
        provider = CoinGeckoProvider()
        
        price = provider.fetch_latest_price("BTC")
        assert price > 0  # Bitcoin should have a price
    
    @pytest.mark.skipif(
        not os.getenv("NEWSAPI_KEY"),
        reason="NEWSAPI_KEY not set"
    )
    def test_newsapi_real_api(self):
        """Test NewsAPI with real API."""
        provider = NewsAPIProvider()
        
        articles = provider.fetch_news("AAPL", limit=5)
        assert isinstance(articles, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
