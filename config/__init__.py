"""
Configuration package.

Use get_settings() for typed, config-driven values. Settings are loaded from
environment variables and optionally from config.config if present.
"""

from config.settings import (
    get_settings,
    TerminalSettings,
    DataSettings,
    BacktestSettings,
    AISettings,
)

__all__ = [
    "get_settings",
    "TerminalSettings",
    "DataSettings",
    "BacktestSettings",
    "AISettings",
]
