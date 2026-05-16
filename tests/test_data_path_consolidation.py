"""
Static guardrails for data-path consolidation.

These checks prevent regression back to ad-hoc DataFetcher imports in core services
that should route through the canonical market data facade.
"""

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_core_services_use_market_data_facade():
    files = [
        "core/backtest_service.py",
        "core/optimizer_service.py",
        "core/stress_test_service.py",
    ]
    for rel in files:
        content = _read(rel)
        assert "market_data_facade" in content, f"{rel} should use canonical facade"
        assert "from core.data_fetcher import DataFetcher" not in content, f"{rel} should not import DataFetcher directly"


def test_market_data_facade_exists_as_canonical_entry():
    content = _read("core/market_data_facade.py")
    assert "def fetch_ohlcv_df(" in content
    assert "def fetch_returns_matrix(" in content
