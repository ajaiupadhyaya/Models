"""
Central configuration for the financial terminal and APIs.

Loads from environment variables (via .env) and optionally from config.config
if present. All runtime settings are typed and accessed through this module.
No magic numbers; feature flags and API keys are config-driven.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Load .env before reading any values
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read from environment; optionally override from config.config if present."""
    value = os.environ.get(key)
    if value is not None:
        return value
    try:
        import config.config as user_config  # type: ignore
        return getattr(user_config, key, None)
    except ImportError:
        pass
    return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float = 0.0) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class DataSettings:
    """Data layer configuration."""
    fred_api_key: Optional[str] = field(default_factory=lambda: _env("FRED_API_KEY"))
    alpha_vantage_api_key: Optional[str] = field(default_factory=lambda: _env("ALPHA_VANTAGE_API_KEY"))
    sample_data_source_default: str = field(default_factory=lambda: _env("SAMPLE_DATA_SOURCE") or "yfinance")

    @property
    def fred_configured(self) -> bool:
        return bool(self.fred_api_key and self.fred_api_key.strip())

    @property
    def alpha_vantage_configured(self) -> bool:
        return bool(self.alpha_vantage_api_key and self.alpha_vantage_api_key.strip())


@dataclass(frozen=True)
class BacktestSettings:
    """Backtesting configuration."""
    use_institutional_default: bool = field(default_factory=lambda: _env_bool("BACKTEST_USE_INSTITUTIONAL_DEFAULT", True))
    default_commission: float = field(default_factory=lambda: _env_float("BACKTEST_DEFAULT_COMMISSION", 0.001))
    default_slippage: float = field(default_factory=lambda: _env_float("BACKTEST_DEFAULT_SLIPPAGE", 0.0005))


@dataclass(frozen=True)
class AISettings:
    """AI / LLM configuration."""
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY"))


@dataclass(frozen=True)
class AuthSettings:
    """Terminal sign-in configuration (env-based, no DB for MVP)."""
    terminal_user: Optional[str] = field(default_factory=lambda: _env("TERMINAL_USER") or "demo")
    terminal_password: Optional[str] = field(default_factory=lambda: _env("TERMINAL_PASSWORD") or "demo")
    auth_secret: str = field(
        default_factory=lambda: _env("AUTH_SECRET") or "change-me-in-production-use-long-random-string"
    )
    token_expire_minutes: int = field(
        default_factory=lambda: int(_env("AUTH_TOKEN_EXPIRE_MINUTES") or "60")
    )

    @property
    def auth_configured(self) -> bool:
        return bool(self.terminal_user and self.terminal_password and self.auth_secret)


@dataclass(frozen=True)
class TerminalSettings:
    """Aggregate settings for the terminal backend."""
    data: DataSettings = field(default_factory=DataSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    ai: AISettings = field(default_factory=AISettings)
    auth: AuthSettings = field(default_factory=AuthSettings)

    @classmethod
    def load(cls) -> TerminalSettings:
        return cls(
            data=DataSettings(),
            backtest=BacktestSettings(),
            ai=AISettings(),
            auth=AuthSettings(),
        )


_settings: Optional[TerminalSettings] = None


def get_settings() -> TerminalSettings:
    """Return the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = TerminalSettings.load()
    return _settings
