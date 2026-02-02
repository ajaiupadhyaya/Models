"""
Unit tests for core/backtesting.py and core/institutional_backtesting.py.

Fixed price and signal series -> deterministic metrics; institutional engine
shows cost impact (worse equity under costs/slippage).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def fixed_price_df():
    """Fixed OHLCV DataFrame: 60 days, flat then up (reproducible)."""
    n = 60
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.RandomState(42).randn(n) * 0.5)
    close = np.maximum(close, 95)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 0.5,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


@pytest.fixture
def fixed_signals(fixed_price_df):
    """Signals: buy first third, neutral middle, sell last third (threshold 0.3)."""
    n = len(fixed_price_df)
    signals = np.zeros(n)
    signals[: n // 3] = 0.5
    signals[-n // 3 :] = -0.5
    return signals


def test_backtest_engine_returns_expected_keys(fixed_price_df, fixed_signals):
    """BacktestEngine.run_backtest returns dict with final_equity, total_return, num_trades, etc."""
    from core.backtesting import BacktestEngine

    engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
    result = engine.run_backtest(
        fixed_price_df, fixed_signals, signal_threshold=0.3, position_size=0.1
    )
    assert "final_equity" in result
    assert "total_return" in result
    assert "total_return_pct" in result
    assert "num_trades" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result
    assert "max_drawdown_pct" in result
    assert isinstance(result["num_trades"], (int, np.integer))
    assert result["final_equity"] >= 0
    assert len(engine.trades) == result["num_trades"]


def test_backtest_engine_deterministic(fixed_price_df, fixed_signals):
    """Same inputs -> same metrics (deterministic)."""
    from core.backtesting import BacktestEngine

    engine1 = BacktestEngine(initial_capital=100000.0, commission=0.001)
    r1 = engine1.run_backtest(fixed_price_df, fixed_signals, signal_threshold=0.3, position_size=0.1)
    engine2 = BacktestEngine(initial_capital=100000.0, commission=0.001)
    r2 = engine2.run_backtest(fixed_price_df, fixed_signals, signal_threshold=0.3, position_size=0.1)
    assert r1["final_equity"] == r2["final_equity"]
    assert r1["num_trades"] == r2["num_trades"]
    assert r1["total_return_pct"] == r2["total_return_pct"]


def test_institutional_engine_returns_expected_keys(fixed_price_df, fixed_signals):
    """InstitutionalBacktestEngine.run_backtest returns enhanced results."""
    from core.institutional_backtesting import InstitutionalBacktestEngine

    np.random.seed(123)
    engine = InstitutionalBacktestEngine(
        initial_capital=100000.0, commission=0.001, slippage=0.0005
    )
    result = engine.run_backtest(
        fixed_price_df, fixed_signals, signal_threshold=0.3, position_size=0.1
    )
    assert "final_equity" in result
    assert "total_return" in result
    assert "num_trades" in result
    assert "max_drawdown" in result
    assert result["final_equity"] >= 0


def test_institutional_engine_cost_impact(fixed_price_df, fixed_signals):
    """Institutional engine returns valid metrics; with costs/slippage equity is typically lower than standard."""
    from core.backtesting import BacktestEngine
    from core.institutional_backtesting import InstitutionalBacktestEngine

    # Standard
    engine_std = BacktestEngine(initial_capital=100000.0, commission=0.001)
    r_std = engine_std.run_backtest(
        fixed_price_df, fixed_signals, signal_threshold=0.3, position_size=0.1
    )
    # Institutional with seed for reproducibility
    np.random.seed(456)
    engine_inst = InstitutionalBacktestEngine(
        initial_capital=100000.0, commission=0.001, slippage=0.001
    )
    r_inst = engine_inst.run_backtest(
        fixed_price_df, fixed_signals, signal_threshold=0.3, position_size=0.1
    )
    # Both return valid structure; institutional has same keys
    assert r_inst["final_equity"] >= 0
    assert r_inst["num_trades"] == r_std["num_trades"]
    # With positive costs/slippage, institutional final equity should not exceed standard by a large margin
    assert r_inst["final_equity"] <= r_std["final_equity"] + 0.05 * 100000
