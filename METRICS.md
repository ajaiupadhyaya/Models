# Analysis Metrics Reference

This document lists each analysis metric used in the system, its formula reference, and the canonical module that implements it. All consumers (APIs, dashboard, integration, model monitor) should call these implementations rather than reimplementing formulas.

## Basic metrics (core/utils.py)

| Metric | Formula / reference | Function |
|--------|---------------------|----------|
| **Returns** | Simple: \( r_t = (P_t - P_{t-1})/P_{t-1} \); Log: \( r_t = \ln(P_t/P_{t-1}) \) | `calculate_returns(prices, method='simple'\|'log')` |
| **Sharpe ratio** | Annualized: \( \sqrt{252} \cdot \frac{\bar{r} - r_f}{\sigma} \) (daily \( r_f/252 \)) | `calculate_sharpe_ratio(returns, risk_free_rate=0.02)` |
| **Sortino ratio** | Annualized excess return / downside deviation (downside std × √252) | `calculate_sortino_ratio(returns, risk_free_rate=0.02)` |
| **Max drawdown** | \( \min_t \frac{C_t - \max_{s\le t} C_s}{\max_{s\le t} C_s} \) where \( C_t = \prod_{s\le t}(1+r_s) \) | `calculate_max_drawdown(returns)` |
| **Max drawdown from equity** | Same formula on normalized equity curve \( E_t/E_0 \) | `calculate_max_drawdown_from_equity(equity_curve)` |
| **Drawdown series** | From returns or equity; used for duration/recovery | `drawdown_series_from_returns(returns)`, `drawdown_series_from_equity(equity_curve)` |
| **VaR** | Historical: percentile of returns at confidence level (e.g. 0.05 = 95% VaR) | `calculate_var(returns, confidence_level=0.05)` |
| **CVaR / Expected shortfall** | Mean of returns ≤ VaR | `calculate_cvar(returns, confidence_level=0.05)` |
| **Annualized return** | \( (1 + \bar{r})^{252} - 1 \) | `annualize_returns(returns, periods_per_year=252)` |
| **Annualized volatility** | \( \sigma \sqrt{252} \) | `annualize_volatility(returns, periods_per_year=252)` |
| **Beta** | \( \mathrm{Cov}(r_a, r_m) / \mathrm{Var}(r_m) \) | `calculate_beta(asset_returns, market_returns)` |

## Risk models (models/risk/var_cvar.py)

| Metric | Method | Usage |
|--------|--------|------|
| **VaR** | Historical, parametric, Monte Carlo | `VaRModel.calculate_var(returns, method=..., confidence_level=0.05)` |
| **CVaR** | Historical, parametric, Monte Carlo | `CVaRModel.calculate_cvar(returns, method=..., confidence_level=0.05)` |

Note: 95% VaR = 5th percentile = `confidence_level=0.05`; 99% VaR = 1st percentile = `confidence_level=0.01`.

## Institutional metrics (models/quant/institutional_grade.py)

| Metric | Formula / reference | Class / method |
|--------|---------------------|----------------|
| **Expected shortfall** | Mean of returns ≤ VaR (same as CVaR) | `AdvancedRiskMetrics.expected_shortfall(returns, confidence=0.05)` |
| **Maximum drawdown (with details)** | Same drawdown formula via core.utils; adds duration, recovery date | `AdvancedRiskMetrics.maximum_drawdown(equity_curve)` |
| **Sortino ratio** | Excess return / annualized downside deviation | `AdvancedRiskMetrics.sortino_ratio(returns, risk_free_rate=0.02)` |
| **Calmar ratio** | Annual return / \|max drawdown\| | `AdvancedRiskMetrics.calmar_ratio(returns, equity_curve)` |
| **Tail ratio** | 95th percentile / \|5th percentile\| | `AdvancedRiskMetrics.tail_ratio(returns, percentile=0.05)` |
| **Information ratio** | Active return / tracking error (vs benchmark) | `AdvancedRiskMetrics.information_ratio(returns, benchmark_returns)` |
| **Bootstrap CI** | Bootstrap samples → percentile CI | `StatisticalValidation.bootstrap_confidence_interval(...)` |
| **Normality test** | Jarque–Bera | `StatisticalValidation.normality_test(returns)` |
| **Stationarity test** | Augmented Dickey–Fuller | `StatisticalValidation.stationarity_test(series)` |
| **Cointegration test** | Engle–Granger | `StatisticalValidation.cointegration_test(series1, series2)` |

## Fundamental metrics (models/fundamental/)

| Area | Module | Notes |
|------|--------|------|
| **Altman Z-Score** | company_analyzer.FundamentalMetrics | Z = 1.2×X1 + 1.4×X2 + 2.3×X3 + 0.6×X4 + 1.0×X5 |
| **Piotroski F-Score** | company_analyzer.FundamentalMetrics | 0–9 from financial statement signals |
| **Financial ratios** | fundamental/ratios.py | Liquidity, leverage, profitability, efficiency, market, DuPont, cash flow |

## Constants

- **Risk-free rate**: Default 0.02 (2% annual); can be overridden via config/env where supported.
- **Periods per year**: 252 for daily equity data.
- **VaR/CVaR confidence**: Use left tail: 0.05 for 95%, 0.01 for 99%.
