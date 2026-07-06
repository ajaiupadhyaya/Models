"""
Backtest service: strategy signal generation + run backtest using real OHLCV from DB/fetcher.
Strategies: sma_cross, rsi_mean_reversion, factor_momentum.
Uses core.db or DataFetcher for price data (no hardcoded data).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV via canonical market data facade."""
    from core.market_data_facade import fetch_ohlcv_df

    return fetch_ohlcv_df(symbol, start_date, end_date)


def _generate_signals(df: pd.DataFrame, strategy: str, **params) -> np.ndarray:
    """Generate trading signals -1 to 1 based on strategy."""
    close = df["Close"]
    n = len(close)
    signals = np.zeros(n)

    if strategy == "sma_cross":
        fast = int(params.get("fast_period", 20))
        slow = int(params.get("slow_period", 50))
        if n < slow:
            return signals
        sma_fast = close.rolling(fast, min_periods=fast).mean()
        sma_slow = close.rolling(slow, min_periods=slow).mean()
        signals = np.where(sma_fast > sma_slow, 1.0, np.where(sma_fast < sma_slow, -1.0, 0.0))
        signals = np.nan_to_num(signals, nan=0.0)

    elif strategy == "rsi_mean_reversion":
        period = int(params.get("rsi_period", 14))
        oversold = int(params.get("oversold", 30))
        overbought = int(params.get("overbought", 70))
        if n < period + 1:
            return signals
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        signals = np.where(rsi < oversold, 1.0, np.where(rsi > overbought, -1.0, 0.0))
        signals = np.nan_to_num(signals, nan=0.0)

    elif strategy == "factor_momentum":
        lookback = int(params.get("lookback", 252))
        if n < lookback:
            return signals
        ret = close.pct_change(lookback)
        ret = ret.fillna(0)
        signals = np.clip(ret.values * 5, -1, 1)

    else:
        signals = np.zeros(n)

    return signals.astype(float)


def _run_backtest_engine(df: pd.DataFrame, signals: np.ndarray, initial_capital: float, commission: float) -> Dict[str, Any]:
    """Run simple backtest, return metrics and equity curve."""
    from core.backtesting import BacktestEngine

    engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
    results = engine.run_backtest(df, signals, signal_threshold=0.0, position_size=0.2)
    equity_curve = getattr(engine, "equity_curve", [])
    trades = getattr(engine, "trades", [])
    eq_dates = df.index[:len(equity_curve)].tolist() if len(equity_curve) <= len(df) else df.index.tolist()
    equity_list = [{"date": (eq_dates[i] if i < len(eq_dates) else df.index[-1]).strftime("%Y-%m-%d"), "equity": float(eq)} for i, eq in enumerate(equity_curve)]
    trades_list = [
        {
            "entry_date": getattr(t, "entry_date", None).strftime("%Y-%m-%d") if getattr(t, "entry_date", None) else "",
            "exit_date": getattr(t, "exit_date", None).strftime("%Y-%m-%d") if getattr(t, "exit_date", None) else "",
            "entry_price": float(getattr(t, "entry_price", 0)),
            "exit_price": float(getattr(t, "exit_price", 0)) if getattr(t, "exit_price", None) else 0,
            "quantity": float(getattr(t, "quantity", 0)),
            "pnl": float(getattr(t, "pnl", 0)) if getattr(t, "pnl", None) else 0,
        }
        for t in trades
    ]
    return {"results": results, "equity_curve": equity_list, "trades": trades_list}


def _compute_benchmark_metrics(strat_returns: pd.Series, bench_returns: pd.Series) -> Dict[str, float]:
    """Compute alpha, beta vs benchmark."""
    aligned = pd.concat([strat_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 2:
        return {"alpha": 0.0, "beta": 0.0}
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    cov = np.cov(y, x)
    if cov[1, 1] != 0:
        beta = cov[0, 1] / cov[1, 1]
    else:
        beta = 0.0
    rf = 0.02 / 252
    alpha = float(y.mean() - rf - beta * (x.mean() - rf))
    return {"alpha": round(alpha * 252 * 100, 2), "beta": round(beta, 4)}


def run_backtest(
    symbol: str,
    strategy: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    commission: float = 0.001,
    **strategy_params,
) -> Dict[str, Any]:
    """
    Run backtest with real OHLCV from DB/fetcher.
    Returns equity_curve, trades, Sharpe, Sortino, CAGR, max_drawdown, win_rate, alpha, beta vs SPY.
    """
    df = _fetch_ohlcv(symbol, start_date, end_date)
    if df is None or df.empty or len(df) < 50:
        return {"error": f"No sufficient price data for {symbol} in date range"}

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return {"error": "No Close price in data"}

    signals = _generate_signals(df, strategy, **strategy_params)
    bt = _run_backtest_engine(df, signals, initial_capital, commission)

    metrics = bt["results"]
    sharpe = float(metrics.get("sharpe_ratio", 0) or 0)
    max_dd = float(metrics.get("max_drawdown", 0) or 0)
    total_return = float(metrics.get("total_return", 0) or 0)
    eq_arr = np.array([e["equity"] for e in bt["equity_curve"]])
    returns = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)
    neg_returns = returns[returns < 0]
    sortino = (np.mean(returns) / np.std(neg_returns) * np.sqrt(252)) if len(neg_returns) > 0 and np.std(neg_returns) > 0 else 0
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    trades = bt["trades"]
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0

    alpha, beta = 0.0, 0.0
    try:
        spy_df = _fetch_ohlcv("SPY", start_date, end_date)
        if spy_df is not None and not spy_df.empty and len(spy_df) > 10:
            strat_eq = pd.Series([e["equity"] for e in bt["equity_curve"]], index=pd.to_datetime([e["date"] for e in bt["equity_curve"]]))
            strat_ret = strat_eq.pct_change().dropna()
            spy_close = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 3]
            bench_ret = spy_close.pct_change().dropna()
            ab = _compute_benchmark_metrics(strat_ret, bench_ret)
            alpha, beta = ab["alpha"], ab["beta"]
    except Exception as e:
        logger.warning("Benchmark alpha/beta failed: %s", e)

    return {
        "symbol": symbol,
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
        "equity_curve": bt["equity_curve"],
        "trades": trades,
        "metrics": {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_return_pct": round(total_return * 100, 2),
            "win_rate_pct": round(win_rate, 2),
            "num_trades": len(trades),
            "alpha_vs_spy": alpha,
            "beta_vs_spy": beta,
        },
    }
