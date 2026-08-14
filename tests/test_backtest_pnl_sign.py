"""Regression tests: realized P&L must ADD to equity, not subtract from it.

core/backtesting.py credited realized trade P&L with `equity -= trade.pnl` at
both realization sites (signal flip, and end-of-run close), while marking to
market with `equity + unrealized_pnl`. The sign was inverted, so a winning
trade reduced final equity and every reported return was backwards.

These tests pin the sign at both sites. Commission is 0 so the only thing
moving equity is the trade P&L itself.
"""

import numpy as np
import pandas as pd
import pytest

from core.backtesting import BacktestEngine


def _ohlcv(closes):
    dates = pd.date_range(start="2024-01-01", periods=len(closes), freq="B")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(len(closes), 1_000_000),
        },
        index=dates,
    )


def test_winning_long_increases_equity_at_end_of_run():
    """Price doubles while held long -> final equity must exceed initial capital."""
    df = _ohlcv(np.linspace(100.0, 200.0, 40))
    signals = np.full(len(df), 1.0)  # stay long throughout

    engine = BacktestEngine(initial_capital=100_000, commission=0.0)
    result = engine.run_backtest(df, signals, signal_threshold=0.3, position_size=0.1)

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    assert trade.pnl > 0, "sanity: a long through a rising market is profitable"

    assert result["total_return"] > 0, (
        f"profitable backtest reported a negative return: {result['total_return']}"
    )
    assert result["final_equity"] > 100_000, (
        f"profitable backtest ended below initial capital: {result['final_equity']}"
    )


def test_losing_long_decreases_equity_at_end_of_run():
    """The inverse must also hold: a losing trade must reduce equity."""
    df = _ohlcv(np.linspace(200.0, 100.0, 40))
    signals = np.full(len(df), 1.0)

    engine = BacktestEngine(initial_capital=100_000, commission=0.0)
    result = engine.run_backtest(df, signals, signal_threshold=0.3, position_size=0.1)

    assert engine.trades[0].pnl < 0
    assert result["final_equity"] < 100_000


def test_winning_trade_closed_by_signal_flip_increases_equity():
    """The signal-flip realization path must use the same sign as end-of-run."""
    # Rise for 20 bars (held long), then flip short for 20 bars.
    closes = np.concatenate([np.linspace(100.0, 200.0, 20), np.full(20, 200.0)])
    signals = np.concatenate([np.full(20, 1.0), np.full(20, -1.0)])

    engine = BacktestEngine(initial_capital=100_000, commission=0.0)
    engine.run_backtest(_ohlcv(closes), signals, signal_threshold=0.3, position_size=0.1)

    long_trades = [t for t in engine.trades if t.position_type == "long"]
    assert long_trades, "expected the long position to be closed by the flip"
    assert long_trades[0].pnl > 0

    # Equity after the flip funded the short at a size derived from equity;
    # the profitable long must have grown the book, not shrunk it.
    assert engine.equity_curve[19] > 100_000


def test_equity_accounting_matches_realized_pnl_exactly():
    """With no commission, final equity == initial capital + sum of realized P&L."""
    df = _ohlcv(np.linspace(100.0, 150.0, 30))
    signals = np.full(len(df), 1.0)

    engine = BacktestEngine(initial_capital=100_000, commission=0.0)
    result = engine.run_backtest(df, signals, signal_threshold=0.3, position_size=0.1)

    realized = sum(t.pnl for t in engine.trades)
    assert result["final_equity"] == pytest.approx(100_000 + realized, rel=1e-9)
