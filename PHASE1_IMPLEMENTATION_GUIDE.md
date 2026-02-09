# Phase 1: Immediate Wins - Implementation Guide

This document provides concrete, copy-paste-ready implementations for Phase 1 integrations.

## Quick Start: Install Phase 1 Libraries

```bash
# Option 1: Install all at once
pip install pmdarima>=2.0.3 tsfresh>=0.19.0 riskfolio-lib>=0.3.0 empyrical>=0.5.5 alphalens-reloaded>=0.4.1 exchange-calendars>=4.2

# Option 2: Use requirements file
pip install -r requirements-quant-phase1.txt
```

## 1. Advanced Time-Series with Auto-ARIMA

### File: `models/timeseries/advanced_ts.py`

```python
"""
Advanced time-series models using pmdarima and tsfresh.
Provides auto-ARIMA, seasonal ARIMA, and automatic feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from tsfresh import extract_features, extract_relevant_features
import warnings
warnings.filterwarnings('ignore')


class AutoArimaForecaster:
    """
    Auto-ARIMA forecaster using pmdarima.
    Automatically detects ARIMA(p,d,q) and SARIMAX(p,d,q)(P,D,Q,m) parameters.
    """
    
    def __init__(self, seasonal: bool = True, m: int = 252, verbose: int = 0):
        """
        Initialize auto-ARIMA forecaster.
        
        Args:
            seasonal: Include seasonality in model
            m: Seasonal period (252 for daily equity data = 1 year)
            verbose: Output verbosity level
        """
        self.seasonal = seasonal
        self.m = m
        self.verbose = verbose
        self.model = None
        self.model_order = None
        self.seasonal_order = None
    
    def fit(self, series: pd.Series, max_p: int = 5, max_q: int = 5) -> Dict:
        """
        Fit auto-ARIMA model to time-series.
        
        Args:
            series: Time-series data
            max_p: Maximum p order
            max_q: Maximum q order
        
        Returns:
            Dictionary with model info and fit statistics
        """
        # Determine differencing order
        d = ndiffs(series, alpha=0.05, max_d=2, test='kpss')
        
        # Fit auto-ARIMA
        self.model = auto_arima(
            series,
            start_p=0,
            start_q=0,
            max_p=max_p,
            max_d=2,
            max_q=max_q,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            trace=self.verbose > 0,
            error_action='ignore',
            suppress_warnings=True,
            maxiter=200,
            return_valid_fits=False
        )
        
        self.model_order = self.model.order
        self.seasonal_order = self.model.seasonal_order if self.seasonal else None
        
        return {
            'order': self.model.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.model.aic(),
            'bic': self.model.bic(),
            'fit_summary': str(self.model.summary())
        }
    
    def forecast(self, steps: int = 20, confidence: float = 0.95) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate forecast and confidence intervals.
        
        Args:
            steps: Number of steps ahead to forecast
            confidence: Confidence level (0.95 for 95% CI)
        
        Returns:
            Tuple of (forecast_series, confidence_intervals_df)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast, conf_int = self.model.get_forecast(
            steps=steps
        ).conf_int(alpha=1-confidence)
        
        return forecast, conf_int
    
    def forecast_with_errors(self, series: pd.Series, steps: int = 20) -> Dict:
        """
        Fit model and produce forecast with error metrics.
        
        Args:
            series: Training series
            steps: Forecast steps
        
        Returns:
            Dict with forecast, CI, and model diagnostics
        """
        fit_info = self.fit(series)
        forecast, conf_int = self.forecast(steps)
        
        return {
            'forecast': forecast,
            'conf_int': conf_int,
            'model_order': fit_info['order'],
            'aic': fit_info['aic'],
            'bic': fit_info['bic'],
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        }


class SeasonalDecomposer:
    """Decompose time-series into trend, seasonal, residual."""
    
    @staticmethod
    def decompose(series: pd.Series, period: int = 252) -> Dict:
        """
        Seasonal decomposition (additive).
        
        Args:
            series: Time-series
            period: Seasonal period
        
        Returns:
            Dict with trend, seasonal, residual, remainder
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }


class TSFeatureExtractor:
    """
    Automatic time-series feature extraction using tsfresh.
    Extracts 700+ features for ML models.
    """
    
    @staticmethod
    def extract_relevant_features(
        df: pd.DataFrame,
        column: str,
        kind: str = 'minimal',
        max_features: int = 50
    ) -> pd.DataFrame:
        """
        Extract relevant features (filter out non-informative ones).
        
        Args:
            df: DataFrame with time-series data
            column: Column name containing time-series
            kind: 'minimal' (~25 features) or 'comprehensive' (700+ features)
            max_features: Max number of features to return
        
        Returns:
            DataFrame with extracted features
        """
        # Prepare data for tsfresh
        ts_data = df[[column]].reset_index(drop=True)
        ts_data['id'] = 1  # Single time-series ID
        ts_data['time'] = range(len(ts_data))
        
        if kind == 'minimal':
            # Extract relevant features with minimal computation
            features = extract_relevant_features(
                ts_data,
                target=None,
                column_id='id',
                column_sort='time',
                kind=kind,
                n_jobs=0,
                show_warnings=False
            )
        else:
            # Extract all features
            features = extract_features(
                ts_data,
                column_id='id',
                column_sort='time',
                parallelization='multiprocessing',
                n_jobs=-1,
                show_warnings=False
            )
        
        return features.head(max_features)
    
    @staticmethod
    def extract_multiple_columns(
        df: pd.DataFrame,
        kind: str = 'minimal'
    ) -> pd.DataFrame:
        """
        Extract features from multiple columns.
        
        Args:
            df: DataFrame with multiple time-series
            kind: 'minimal' or 'comprehensive'
        
        Returns:
            DataFrame with feature matrix
        """
        all_features = {}
        
        for col in df.columns:
            ts_data = df[[col]].reset_index(drop=True)
            ts_data['id'] = col
            ts_data['time'] = range(len(ts_data))
            
            if kind == 'minimal':
                features = extract_relevant_features(
                    ts_data,
                    target=None,
                    column_id='id',
                    column_sort='time',
                    kind=kind,
                    n_jobs=0,
                    show_warnings=False
                )
            else:
                features = extract_features(
                    ts_data,
                    column_id='id',
                    column_sort='time',
                    n_jobs=0,
                    show_warnings=False
                )
            
            all_features[col] = features
        
        # Concatenate with column names as prefix
        result = pd.concat(
            [features.add_prefix(f'{col}_') for col, features in all_features.items()],
            axis=1
        )
        
        return result


class UnivariateForecaster:
    """Wrapper for simple univariate forecasting."""
    
    def __init__(self, series: pd.Series, seasonal: bool = True):
        self.series = series
        self.forecaster = AutoArimaForecaster(seasonal=seasonal)
        self.fit_info = None
    
    def fit_and_forecast(self, steps: int = 20) -> Dict:
        """Convenience method: fit and forecast in one call."""
        self.fit_info = self.forecaster.fit(self.series)
        forecast, conf_int = self.forecaster.forecast(steps=steps)
        
        return {
            'forecast': forecast,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1],
            'aic': self.fit_info['aic'],
            'model_order': self.fit_info['order']
        }
```

### Integration: Add to `api/predictions_api.py`

```python
from models.timeseries.advanced_ts import AutoArimaForecaster, UnivariateForecaster

@router.get("/forecast-arima/{ticker}")
async def forecast_arima(
    ticker: str = Query(..., description="Stock symbol"),
    steps: int = Query(20, min=1, max=100, description="Forecast horizon (days)"),
    seasonal: bool = Query(True, description="Include seasonality")
) -> Dict[str, Any]:
    """
    Auto-ARIMA forecast for stock price momentum/returns.
    Uses pmdarima to auto-select (p,d,q)(P,D,Q,m) parameters.
    
    Response includes forecast, 95% confidence intervals, and model fit metrics.
    """
    try:
        import yfinance as yf
        from models.timeseries.advanced_ts import UnivariateForecaster
        
        # Fetch daily returns
        data = yf.download(ticker, period='2y', interval='1d', progress=False)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        
        # Use returns instead of prices
        returns = data['Close'].pct_change().dropna()
        
        # Forecast
        forecaster = UnivariateForecaster(returns, seasonal=seasonal)
        result = forecaster.fit_and_forecast(steps=steps)
        
        return {
            'symbol': ticker,
            'forecast_type': 'ARIMA(auto)',
            'model_order': str(result['model_order']),
            'forecast': result['forecast'].to_dict(),
            'lower_bound': result['lower'].to_dict(),
            'upper_bound': result['upper'].to_dict(),
            'aic': float(result['aic']),
            'note': 'Forecast is for returns; add to last price to get predicted price'
        }
    
    except Exception as e:
        logger.error(f"ARIMA forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 2. Advanced Portfolio Optimization with CVaR

### File: `models/portfolio/advanced_optimization.py`

```python
"""
Advanced portfolio optimization using riskfolio-lib and empyrical.
Includes CVaR, entropy pooling, and enhanced risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import riskfolio as rp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class CvaROptimizer:
    """
    Conditional Value-at-Risk (CVaR) portfolio optimization.
    More sophisticated than mean-variance for tail-risk management.
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize CVaR optimizer.
        
        Args:
            returns_df: Asset returns (columns = assets, rows = dates)
        """
        self.returns = returns_df
        self.weights = None
        self.portfolio = rp.Portfolio(returns=returns_df)
        self.portfolio.assets_stats(method_mu='hist', method_cov='hist')
    
    def optimize_cvar(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize portfolio using CVaR as risk measure.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
        
        Returns:
            Dict with weights, CVaR, return, volatility
        """
        self.portfolio.optimization(
            model='CVaR',
            rm='CVaR',
            obj='Sharpe',
            rf=risk_free_rate,
            l=0,
            hist=True
        )
        
        self.weights = self.portfolio.allocation.ravel()
        
        return {
            'weights': {asset: float(w) for asset, w in zip(self.returns.columns, self.weights)},
            'expected_return': float(self.portfolio.ret),
            'cvar_95': float(self.portfolio.cvar),
            'volatility': float(self.portfolio.risk),
            'sharpe_ratio': float(self.portfolio.sharpe)
        }
    
    def optimize_min_cvar(self) -> Dict:
        """Minimize CVaR (conservative, tail-risk focus)."""
        self.portfolio.optimization(
            model='CVaR',
            rm='CVaR',
            obj='MinRisk',
            hist=True
        )
        
        self.weights = self.portfolio.allocation.ravel()
        
        return {
            'weights': {asset: float(w) for asset, w in zip(self.returns.columns, self.weights)},
            'cvar_95': float(self.portfolio.cvar),
            'volatility': float(self.portfolio.risk),
            'objective': 'Minimum CVaR'
        }
    
    def efficient_frontier_cvar(self, points: int = 20) -> Dict:
        """
        Generate CVaR-based efficient frontier.
        
        Args:
            points: Number of points on frontier
        
        Returns:
            Dict with frontier points and weights
        """
        mu = self.portfolio.mu
        cov = self.portfolio.cov
        
        # Generate returns from min CVaR to max return
        targets = np.linspace(mu.min(), mu.max(), points)
        frontier_points = []
        frontier_weights = []
        
        for target_return in targets:
            self.portfolio.optimization(
                model='CVaR',
                rm='CVaR',
                obj='MinRisk',
                constraints={'return': target_return},
                hist=True
            )
            
            frontier_points.append({
                'return': float(self.portfolio.ret),
                'cvar': float(self.portfolio.cvar),
                'volatility': float(self.portfolio.risk)
            })
            frontier_weights.append(self.portfolio.allocation.ravel())
        
        return {
            'frontier': frontier_points,
            'weights': [
                {asset: float(w) for asset, w in zip(self.returns.columns, weights)}
                for weights in frontier_weights
            ]
        }


class RiskParityOptimizer:
    """
    Risk parity portfolio: equal contribution to portfolio risk.
    Good for diversification when vol differs significantly across assets.
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        self.returns = returns_df
        self.portfolio = rp.Portfolio(returns=returns_df)
        self.portfolio.assets_stats(method_mu='hist', method_cov='hist')
    
    def optimize_risk_parity(self) -> Dict:
        """
        Optimize for equal risk contribution.
        """
        self.portfolio.optimization(
            model='RiskParity',
            rm='MV',
            rf=0.02
        )
        
        return {
            'weights': {asset: float(w) for asset, w in zip(self.returns.columns, self.portfolio.allocation.ravel())},
            'expected_return': float(self.portfolio.ret),
            'volatility': float(self.portfolio.risk),
            'sharpe_ratio': float(self.portfolio.sharpe),
            'method': 'Risk Parity (Equal Risk Contribution)'
        }


class EnhancedPortfolioMetrics:
    """Calculate comprehensive portfolio performance metrics using empyrical."""
    
    @staticmethod
    def calculate_metrics(returns_series: pd.Series, risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate portfolio metrics.
        
        Args:
            returns_series: Daily portfolio returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Dict with comprehensive metrics
        """
        from empyrical import (
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            stability_of_timeseries,
            max_drawdown,
            capture_ratios
        )
        
        metrics = {
            'total_return': float((1 + returns_series).prod() - 1),
            'annual_return': float(annual_return(returns_series)),
            'annual_volatility': float(annual_volatility(returns_series)),
            'sharpe_ratio': float(sharpe_ratio(returns_series, risk_free_rate)),
            'sortino_ratio': float(sortino_ratio(returns_series, risk_free_rate)),
            'calmar_ratio': float(calmar_ratio(returns_series)),
            'max_drawdown': float(max_drawdown(returns_series)),
            'stability': float(stability_of_timeseries(returns_series))
        }
        
        return metrics
```

### Integration: Add to `api/risk_api.py`

```python
from models.portfolio.advanced_optimization import CvaROptimizer, RiskParityOptimizer, EnhancedPortfolioMetrics

@router.get("/optimize-cvar")
async def optimize_cvar_portfolio(
    symbols: str = Query("AAPL,MSFT,GOOGL,AMZN", description="Comma-separated symbols"),
    period: str = Query("2y", description="History period"),
    risk_free_rate: float = Query(0.02, description="Risk-free rate")
) -> Dict[str, Any]:
    """
    CVaR-based portfolio optimization for tail-risk awareness.
    Better than mean-variance for investors concerned about downside tail.
    """
    try:
        import yfinance as yf
        
        # Download returns
        sym_list = [s.strip().upper() for s in symbols.split(",")]
        data = yf.download(sym_list, period=period, interval='1d', progress=False)['Adj Close']
        
        if data.empty or len(sym_list) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 symbols with data")
        
        returns = data.pct_change().dropna()
        
        # Optimize
        optimizer = CvaROptimizer(returns)
        result = optimizer.optimize_cvar(risk_free_rate=risk_free_rate)
        
        return {
            'symbols': sym_list,
            'optimization_method': 'CVaR (Conditional Value-at-Risk)',
            'period': period,
            'weights': result['weights'],
            'expected_annual_return': f"{result['expected_return']*100:.2f}%",
            'cvar_95': f"{result['cvar_95']*100:.2f}%",
            'volatility': f"{result['volatility']*100:.2f}%",
            'sharpe_ratio': f"{result['sharpe_ratio']:.2f}",
            'note': 'CVaR is the average loss in worst 5% of scenarios'
        }
    
    except Exception as e:
        logger.error(f"CVaR optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 3. Trading Calendar Integration

### File: `core/trading_calendar.py`

```python
"""
Trading calendar awareness - avoid backtesting on non-trading days.
Uses exchange-calendars for NYSE, NASDAQ, LSE, etc.
"""

import pandas as pd
from exchange_calendars import get_calendar
from typing import List, Optional


class TradingCalendar:
    """Multi-exchange trading calendar."""
    
    SUPPORTED_EXCHANGES = {
        'NYSE': 'New York Stock Exchange',
        'NASDAQ': 'NASDAQ',
        'LSE': 'London Stock Exchange',
        'TSE': 'Tokyo Stock Exchange',
        'HK': 'Hong Kong Stock Exchange'
    }
    
    def __init__(self, exchange: str = 'NYSE'):
        """
        Initialize trading calendar.
        
        Args:
            exchange: Exchange code (NYSE, NASDAQ, etc.)
        """
        if exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Unknown exchange: {exchange}. Supported: {list(self.SUPPORTED_EXCHANGES.keys())}")
        
        self.exchange = exchange
        self.calendar = get_calendar(exchange)
    
    def trading_days(self, start: str, end: str) -> pd.DatetimeIndex:
        """
        Get all trading days in range.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            DatetimeIndex of trading days
        """
        return self.calendar.sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end)
        )
    
    def is_trading_day(self, date: str) -> bool:
        """Check if date is a trading day."""
        return self.calendar.is_trading_day(pd.Timestamp(date))
    
    def next_trading_day(self, date: str) -> str:
        """Get next trading day after date."""
        ts = pd.Timestamp(date)
        sessions = self.calendar.sessions
        
        if ts >= sessions[-1]:
            return str(sessions[-1].date())
        
        idx = sessions.get_loc(ts, method='nearest')
        if idx < len(sessions) - 1:
            return str(sessions[idx + 1].date())
        return str(sessions[-1].date())
    
    def previous_trading_day(self, date: str) -> str:
        """Get previous trading day before date."""
        ts = pd.Timestamp(date)
        sessions = self.calendar.sessions
        
        if ts <= sessions[0]:
            return str(sessions[0].date())
        
        idx = sessions.get_loc(ts, method='nearest')
        if idx > 0:
            return str(sessions[idx - 1].date())
        return str(sessions[0].date())
    
    def trading_days_between(self, start: str, end: str, count: int = 1) -> List[str]:
        """Get N trading days in range."""
        days = self.trading_days(start, end)
        return [str(d.date()) for d in days[:count]]
```

### Update `core/backtesting.py` to use TradingCalendar

```python
from core.trading_calendar import TradingCalendar

class BacktestEngine:
    def __init__(self, exchange: str = 'NYSE', **kwargs):
        """Add trading calendar awareness."""
        self.calendar = TradingCalendar(exchange)
        # ... rest of init
    
    def run_backtest(self, df: pd.DataFrame, signals: pd.Series, **kwargs):
        """Filter to trading days only."""
        # Ensure only trading days
        trading_days = self.calendar.trading_days(
            str(df.index[0].date()),
            str(df.index[-1].date())
        )
        
        # Align data to trading days
        df_clean = df.loc[df.index.isin(trading_days)]
        signals_clean = signals.loc[signals.index.isin(trading_days)]
        
        # Run backtest on clean data
        return super().run_backtest(df_clean, signals_clean, **kwargs)
```

---

## Testing Phase 1 Implementations

### Test Script: `test_phase1_integration.py`

```python
"""
Test Phase 1 implementations.
Run: python test_phase1_integration.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from models.timeseries.advanced_ts import AutoArimaForecaster, FeatureExtractor
from models.portfolio.advanced_optimization import CvaROptimizer, EnhancedPortfolioMetrics
from core.trading_calendar import TradingCalendar


def test_arima_forecast():
    """Test auto-ARIMA forecasting."""
    print("\n✓ Testing AutoArimaForecaster...")
    
    # Get data
    data = yf.download('AAPL', period='3y', progress=False)
    returns = data['Close'].pct_change().dropna()
    
    # Forecast
    forecaster = AutoArimaForecaster()
    result = forecaster.fit(returns)
    
    print(f"  Model order: {result['order']}")
    print(f"  AIC: {result['aic']:.2f}")
    
    forecast, ci = forecaster.forecast(steps=20)
    print(f"  20-step forecast shape: {forecast.shape}")
    print(f"  ✅ ARIMA test passed")


def test_cvar_optimization():
    """Test CVaR portfolio optimization."""
    print("\n✓ Testing CvaROptimizer...")
    
    # Get data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    data = yf.download(tickers, period='2y', progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    # Optimize
    optimizer = CvaROptimizer(returns)
    result = optimizer.optimize_cvar()
    
    print(f"  Weights: {result['weights']}")
    print(f"  Expected return: {result['expected_return']*100:.2f}%")
    print(f"  Sharpe ratio: {result['sharpe_ratio']:.2f}")
    print(f"  ✅ CVaR test passed")


def test_trading_calendar():
    """Test trading calendar."""
    print("\n✓ Testing TradingCalendar...")
    
    cal = TradingCalendar('NYSE')
    
    # Test trading day check
    assert not cal.is_trading_day('2024-01-01')  # New Year's Day
    assert cal.is_trading_day('2024-01-02')       # Trading day
    
    # Test next trading day
    next_day = cal.next_trading_day('2024-01-01')
    print(f"  Next trading day after 2024-01-01: {next_day}")
    
    # Test trading days range
    days = cal.trading_days('2024-01-01', '2024-01-31')
    print(f"  Trading days in Jan 2024: {len(days)}")
    
    print(f"  ✅ Trading calendar test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 1 Integration Tests")
    print("=" * 60)
    
    test_arima_forecast()
    test_cvar_optimization()
    test_trading_calendar()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 1 tests passed!")
    print("=" * 60)
```

Run with:
```bash
python test_phase1_integration.py
```

---

## Summary: What You've Implemented

| Feature | Library | File | API Endpoint |
|---------|---------|------|--------------|
| Auto-ARIMA forecasting | pmdarima | `models/timeseries/advanced_ts.py` | `GET /api/v1/forecast-arima/{ticker}` |
| CVaR optimization | riskfolio-lib | `models/portfolio/advanced_optimization.py` | `GET /api/v1/portfolio/optimize-cvar` |
| Trading calendar | exchange-calendars | `core/trading_calendar.py` | Internal (used in backtesting) |
| Enhanced metrics | empyrical | `models/portfolio/advanced_optimization.py` | Part of portfolio APIs |

---

## Next Steps

1. Install Phase 1 libraries: `pip install pmdarima tsfresh riskfolio-lib empyrical alphalens-reloaded exchange-calendars`
2. Copy implementation files to your project
3. Update API endpoints in `api/predictions_api.py` and `api/risk_api.py`
4. Test with `test_phase1_integration.py`
5. Update API documentation with new endpoints
6. Deploy and monitor performance

---

**Document Version:** 1.0  
**Created:** Feb 9, 2026
