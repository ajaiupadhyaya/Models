"""
Stress testing using real historical OHLCV during crisis windows.
No hardcoded shock values for user tickers - pulls actual returns from DB/fetcher.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Crisis scenario date windows (peak-to-trough)
CRISIS_SCENARIOS = {
    "financial_crisis_2008": {
        "name": "2008 Financial Crisis",
        "start": "2008-09-01",
        "end": "2009-03-09",
        "description": "Peak-to-trough Sep 2008–Mar 2009",
    },
    "covid_crash_2020": {
        "name": "COVID Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "Feb–Mar 2020 pandemic panic",
    },
    "dotcom_bust": {
        "name": "Dot-com Bust",
        "start": "2000-03-10",
        "end": "2002-10-09",
        "description": "Mar 2000–Oct 2002",
    },
    "rate_shock_2022": {
        "name": "2022 Rate Shock",
        "start": "2022-01-03",
        "end": "2022-10-12",
        "description": "2022 Fed tightening selloff",
    },
}


def _fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> Optional[Any]:
    """Fetch OHLCV through canonical market data facade."""
    from core.market_data_facade import fetch_ohlcv_df

    return fetch_ohlcv_df(symbol, start_date, end_date)


def run_stress_test(
    tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Apply crisis scenarios using real historical returns from OHLCV.
    Returns portfolio drawdown per scenario, worst single-day loss, recovery time (days).
    """
    import pandas as pd

    tickers = [t.upper() for t in tickers]
    if not tickers:
        return {"error": "No tickers provided"}

    n = len(tickers)
    w = weights or {s: 1.0 / n for s in tickers}
    w_sum = sum(w.get(s, 1.0 / n) for s in tickers)
    if abs(w_sum - 1.0) > 0.01:
        w = {s: w.get(s, 1.0 / n) / w_sum for s in tickers}
    else:
        w = {s: w.get(s, 1.0 / n) for s in tickers}

    scenarios_out = []

    for scenario_id, meta in CRISIS_SCENARIOS.items():
        start = meta["start"]
        end = meta["end"]

        prices = {}
        for sym in tickers:
            df = _fetch_ohlcv(sym, start, end)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    ser = df.xs("Close", axis=1, level=1).iloc[:, 0]
                except (KeyError, IndexError):
                    continue
            else:
                ser = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
            prices[sym] = ser.dropna().rename(sym)

        if not prices:
            scenarios_out.append({
                "scenario_id": scenario_id,
                "name": meta["name"],
                "description": meta["description"],
                "start": start,
                "end": end,
                "portfolio_drawdown_pct": None,
                "worst_single_day_loss_pct": None,
                "recovery_days": None,
                "error": "No price data for any ticker in this period",
            })
            continue

        # Align to common index
        aligned = pd.DataFrame(prices).dropna()
        if aligned.empty or len(aligned) < 2:
            scenarios_out.append({
                "scenario_id": scenario_id,
                "name": meta["name"],
                "description": meta["description"],
                "start": start,
                "end": end,
                "portfolio_drawdown_pct": None,
                "worst_single_day_loss_pct": None,
                "recovery_days": None,
                "error": "Insufficient aligned data",
            })
            continue

        # Portfolio returns
        ret = aligned.pct_change().dropna()
        w_sub = {s: w.get(s, 1.0 / n) for s in aligned.columns}
        w_sum = sum(w_sub.values())
        w_ser = pd.Series({s: v / w_sum for s, v in w_sub.items()})
        port_ret = (ret * w_ser).sum(axis=1)

        # Cumulative wealth
        cum = (1 + port_ret).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = float(dd.min()) if len(dd) > 0 else 0
        worst_day = float(port_ret.min()) if len(port_ret) > 0 else 0

        # Recovery: first day after trough where cum >= previous peak
        trough_idx = cum.idxmin()
        cum.min()
        after = cum.loc[trough_idx:]
        recovered = after[after >= peak.loc[trough_idx]]
        recovery_days = None
        if len(recovered) > 0:
            first_recovery = recovered.index[0]
            if hasattr(first_recovery, "date"):
                delta = (first_recovery - trough_idx).days if hasattr(trough_idx, "date") else None
            else:
                delta = (pd.Timestamp(first_recovery) - pd.Timestamp(trough_idx)).days
            recovery_days = delta

        scenarios_out.append({
            "scenario_id": scenario_id,
            "name": meta["name"],
            "description": meta["description"],
            "start": start,
            "end": end,
            "portfolio_drawdown_pct": round(max_dd * 100, 2),
            "worst_single_day_loss_pct": round(worst_day * 100, 2),
            "recovery_days": recovery_days,
        })

    return {
        "tickers": tickers,
        "weights": w,
        "scenarios": scenarios_out,
    }
