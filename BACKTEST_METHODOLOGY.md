# Backtest Methodology

This document describes assumptions, slippage/cost models, and how to reproduce backtest results.

## Data requirements

- **OHLCV**: Backtest engines expect a DataFrame with columns `Open`, `High`, `Low`, `Close`, `Volume` and a time-index (e.g. `DatetimeIndex`).
- **Point-in-time**: Results are intended to be reproducible with the same data; no lookahead is used. In production, use point-in-time datasets to avoid survivorship and lookahead bias.
- **Frequency**: Default is daily (252 trading days per year for annualization).

## Engines

### BacktestEngine (core/backtesting.py)

- **Initial capital**: Configurable (default 100,000).
- **Commission**: Per-trade commission (e.g. 0.001 = 0.1%).
- **Position sizing**: Fraction of capital per signal (e.g. `position_size=0.1` = 10%).
- **Signal threshold**: Signals above `signal_threshold` trigger long, below `-signal_threshold` trigger short; otherwise flat.
- **Execution**: Assumes fill at `Close` of the bar where the signal occurs (no intra-bar modeling).
- **Metrics**: Uses `core.utils` for Sharpe ratio and max drawdown (single source of truth).

### InstitutionalBacktestEngine (core/institutional_backtesting.py)

- Extends `BacktestEngine` with:
  - **Slippage**: Applied to entry/exit prices (e.g. 0.0005 = 5 bps).
  - **Market impact**: Optional cost as a function of trade size, daily volume, and volatility (see `models/quant/institutional_grade.py` `TransactionCostModel`).
  - **Advanced risk metrics**: Sortino, Calmar, expected shortfall, tail ratio, normality test (via `AdvancedRiskMetrics` and `StatisticalValidation`).
- With positive commission and slippage, final equity is typically lower than the standard engine for the same signals.

## Reproducibility

- **Seeds**: Use fixed seeds for any randomness (e.g. `np.random.seed(42)` before running) when comparing runs.
- **Inputs**: Same `DataFrame` (OHLCV) and same signal array produce deterministic metrics for `BacktestEngine`. `InstitutionalBacktestEngine` may use randomness for market impact; set seed for reproducibility.
- **Config**: Document or pass the same `initial_capital`, `commission`, `slippage`, `signal_threshold`, and `position_size` when reproducing.

## Metrics

- **Total return**: (final_equity - initial_capital) / initial_capital.
- **Sharpe ratio**: Annualized (√252 × excess return / volatility); risk-free rate default 2% (see `core.utils.calculate_sharpe_ratio`).
- **Max drawdown**: From cumulative equity; see `METRICS.md` and `core.utils.calculate_max_drawdown`.

## Tests

- `tests/test_core_backtesting.py`: Fixed price and signal series → deterministic metrics; institutional engine shows cost impact (equity ≤ standard + tolerance).
