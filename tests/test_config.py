"""
Unit tests for config/settings.py.

Tests DataSettings, BacktestSettings, AISettings, TerminalSettings,
and get_settings() with minimal env coupling (explicit dataclass construction).
"""

import os
import pytest


@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """Reset the settings singleton so each test gets a fresh load."""
    import config.settings as mod
    mod._settings = None
    yield
    mod._settings = None


def test_data_settings_defaults():
    """DataSettings with no env: default sample_data_source, no keys."""
    from config.settings import DataSettings, _env

    # Construct with explicit values (avoids env)
    settings = DataSettings(
        fred_api_key=None,
        alpha_vantage_api_key=None,
        sample_data_source_default="yfinance",
    )
    assert settings.fred_api_key is None
    assert settings.alpha_vantage_api_key is None
    assert settings.sample_data_source_default == "yfinance"
    assert settings.fred_configured is False
    assert settings.alpha_vantage_configured is False


def test_data_settings_configured():
    """DataSettings with keys: fred_configured and alpha_vantage_configured true."""
    from config.settings import DataSettings

    settings = DataSettings(
        fred_api_key="test-fred-key",
        alpha_vantage_api_key="test-av-key",
        sample_data_source_default="data_fetcher",
    )
    assert settings.fred_configured is True
    assert settings.alpha_vantage_configured is True
    assert settings.sample_data_source_default == "data_fetcher"


def test_backtest_settings_defaults():
    """BacktestSettings: use_institutional_default, commission, slippage."""
    from config.settings import BacktestSettings

    settings = BacktestSettings(
        use_institutional_default=True,
        default_commission=0.001,
        default_slippage=0.0005,
    )
    assert settings.use_institutional_default is True
    assert settings.default_commission == 0.001
    assert settings.default_slippage == 0.0005


def test_ai_settings():
    """AISettings holds optional openai_api_key."""
    from config.settings import AISettings

    settings = AISettings(openai_api_key="sk-test")
    assert settings.openai_api_key == "sk-test"

    empty = AISettings(openai_api_key=None)
    assert empty.openai_api_key is None


def test_terminal_settings_load_structure():
    """TerminalSettings.load() returns instance with data, backtest, ai."""
    from config.settings import TerminalSettings, get_settings

    settings = get_settings()
    assert isinstance(settings, TerminalSettings)
    assert hasattr(settings, "data")
    assert hasattr(settings, "backtest")
    assert hasattr(settings, "ai")
    assert settings.data is not None
    assert settings.backtest is not None
    assert settings.ai is not None


def test_get_settings_singleton():
    """get_settings() returns the same instance on repeated calls."""
    from config.settings import get_settings

    a = get_settings()
    b = get_settings()
    assert a is b


def test_settings_from_env(monkeypatch):
    """When env vars are set, load() picks them up (no config.config)."""
    from config.settings import TerminalSettings

    monkeypatch.setenv("FRED_API_KEY", "env-fred")
    monkeypatch.setenv("BACKTEST_USE_INSTITUTIONAL_DEFAULT", "false")
    monkeypatch.setenv("SAMPLE_DATA_SOURCE", "data_fetcher")

    # Force reload by clearing singleton
    import config.settings as mod
    mod._settings = None

    settings = mod.get_settings()
    assert settings.data.fred_api_key == "env-fred"
    assert settings.data.sample_data_source_default == "data_fetcher"
    assert settings.backtest.use_institutional_default is False


def test_env_bool_edge_cases(monkeypatch):
    """_env_bool: true/1/yes/on and false/0/no/off; default when unset."""
    import config.settings as mod
    mod._settings = None
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv("BACKTEST_USE_INSTITUTIONAL_DEFAULT", val)
        mod._settings = None
        s = mod.get_settings()
        assert s.backtest.use_institutional_default is True
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("BACKTEST_USE_INSTITUTIONAL_DEFAULT", val)
        mod._settings = None
        s = mod.get_settings()
        assert s.backtest.use_institutional_default is False
    monkeypatch.delenv("BACKTEST_USE_INSTITUTIONAL_DEFAULT", raising=False)
    mod._settings = None
    s = mod.get_settings()
    assert s.backtest.use_institutional_default is True  # default True per settings


def test_env_float_edge_cases(monkeypatch):
    """_env_float: valid float and invalid falls back to default."""
    import config.settings as mod
    mod._settings = None
    monkeypatch.setenv("BACKTEST_DEFAULT_COMMISSION", "0.002")
    mod._settings = None
    s = mod.get_settings()
    assert s.backtest.default_commission == 0.002
    monkeypatch.setenv("BACKTEST_DEFAULT_COMMISSION", "not_a_number")
    mod._settings = None
    s = mod.get_settings()
    assert s.backtest.default_commission == 0.001  # default
