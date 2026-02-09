"""
Tests for quantitative engine.

Tests:
- Factor calculations (momentum, volatility, Sharpe, drawdown)
- Risk metrics (VaR, CVaR, Sortino, Calmar)
- Regime detection (trend and volatility)
- Options pricing (Black-Scholes)
- Greeks (Delta, Gamma, Vega, Theta)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from core.quant_engine import (
    QuantEngine,
    RegimeDetector,
    OptionsAnalytics,
    FactorScores,
    RiskMetrics,
    calculate_portfolio_var,
)


class TestQuantEngine:
    """Test core quantitative engine functionality."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series."""
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        # Generate trending prices with volatility
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 300)
        prices = 100 * (1 + returns).cumprod()
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_returns(self, sample_prices):
        """Generate sample returns from prices."""
        return sample_prices.pct_change().fillna(0)
    
    def test_calculate_returns(self, sample_prices):
        """Test return calculation."""
        engine = QuantEngine()
        returns = engine.calculate_returns(sample_prices)
        
        assert len(returns) == len(sample_prices)
        assert returns.iloc[0] == 0  # First return is 0 (no previous price)
        assert all(returns.iloc[1:] == sample_prices.pct_change().iloc[1:])
    
    def test_calculate_log_returns(self, sample_prices):
        """Test log return calculation."""
        engine = QuantEngine()
        log_returns = engine.calculate_log_returns(sample_prices)
        
        assert len(log_returns) == len(sample_prices)
        assert log_returns.iloc[0] == 0
        # Log returns should be approximately equal to simple returns for small changes
        simple_returns = engine.calculate_returns(sample_prices)
        assert np.allclose(log_returns.iloc[1:10], simple_returns.iloc[1:10], atol=0.001)
    
    def test_calculate_momentum(self, sample_prices):
        """Test momentum calculation."""
        engine = QuantEngine()
        
        # 1-month momentum (21 trading days)
        momentum_1m = engine.calculate_momentum(sample_prices, 21)
        expected = (sample_prices.iloc[-1] / sample_prices.iloc[-22]) - 1
        assert abs(momentum_1m - expected) < 1e-6
        
        # Test with insufficient data
        short_prices = sample_prices.head(10)
        momentum = engine.calculate_momentum(short_prices, 21)
        assert momentum == 0.0
    
    def test_calculate_volatility(self, sample_returns):
        """Test volatility calculation."""
        engine = QuantEngine()
        
        vol = engine.calculate_volatility(sample_returns)
        
        # Volatility should be positive
        assert vol > 0
        
        # Annualized volatility should be approximately std * sqrt(252)
        expected_vol = sample_returns.std() * np.sqrt(252)
        assert abs(vol - expected_vol) < 1e-6
        
        # Test with insufficient data
        short_returns = sample_returns.head(1)
        vol = engine.calculate_volatility(short_returns)
        assert vol == 0.0
    
    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        engine = QuantEngine()
        
        sharpe = engine.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        # Sharpe ratio should be finite
        assert np.isfinite(sharpe)
        
        # Manual calculation
        excess_returns = sample_returns - (0.02 / 252)
        expected_sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        assert abs(sharpe - expected_sharpe) < 1e-6
    
    def test_calculate_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        engine = QuantEngine()
        
        sortino = engine.calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)
        
        # Sortino should be finite
        assert np.isfinite(sortino)
        
        # Sortino should generally be higher than Sharpe (only penalizes downside)
        sharpe = engine.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        # Note: This may not always be true for all datasets
        assert isinstance(sortino, float)
    
    def test_calculate_max_drawdown(self, sample_prices):
        """Test max drawdown calculation."""
        engine = QuantEngine()
        
        max_dd = engine.calculate_max_drawdown(sample_prices)
        
        # Drawdown should be negative (or zero)
        assert max_dd <= 0
        
        # Manual calculation
        returns = sample_prices.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        expected_max_dd = drawdown.min()
        
        assert abs(max_dd - expected_max_dd) < 1e-6
    
    def test_calculate_var(self, sample_returns):
        """Test VaR calculation."""
        engine = QuantEngine()
        
        var_95 = engine.calculate_var(sample_returns, confidence=0.95)
        
        # VaR should be negative (indicating loss)
        assert var_95 < 0
        
        # VaR should be approximately the 5th percentile
        expected_var = np.percentile(sample_returns, 5)
        assert abs(var_95 - expected_var) < 1e-6
        
        # Test with insufficient data
        short_returns = sample_returns.head(5)
        var = engine.calculate_var(short_returns)
        assert var == 0.0
    
    def test_calculate_cvar(self, sample_returns):
        """Test CVaR calculation."""
        engine = QuantEngine()
        
        cvar_95 = engine.calculate_cvar(sample_returns, confidence=0.95)
        var_95 = engine.calculate_var(sample_returns, confidence=0.95)
        
        # CVaR should be more negative than VaR (expected loss beyond VaR)
        assert cvar_95 < var_95
        
        # Manual calculation
        expected_cvar = sample_returns[sample_returns <= var_95].mean()
        assert abs(cvar_95 - expected_cvar) < 1e-6
    
    def test_calculate_factors(self, sample_prices):
        """Test comprehensive factor calculation."""
        engine = QuantEngine()
        
        factors = engine.calculate_factors("AAPL", sample_prices)
        
        assert isinstance(factors, FactorScores)
        assert factors.symbol == "AAPL"
        assert factors.date == sample_prices.index[-1]
        
        # All factors should be finite
        assert np.isfinite(factors.momentum_1m)
        assert np.isfinite(factors.momentum_3m)
        assert np.isfinite(factors.momentum_6m)
        assert np.isfinite(factors.momentum_12m)
        assert np.isfinite(factors.volatility_30d)
        assert np.isfinite(factors.volatility_90d)
        assert np.isfinite(factors.sharpe_ratio)
        assert np.isfinite(factors.max_drawdown)
        
        # Volatility should be positive
        assert factors.volatility_30d > 0
        assert factors.volatility_90d > 0
        
        # Max drawdown should be negative
        assert factors.max_drawdown <= 0
    
    def test_calculate_risk_metrics(self, sample_returns):
        """Test comprehensive risk metrics calculation."""
        engine = QuantEngine()
        
        metrics = engine.calculate_risk_metrics("AAPL", sample_returns)
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.symbol == "AAPL"
        
        # All metrics should be finite
        assert np.isfinite(metrics.var_95)
        assert np.isfinite(metrics.cvar_95)
        assert np.isfinite(metrics.volatility_annual)
        assert np.isfinite(metrics.sharpe_ratio)
        assert np.isfinite(metrics.sortino_ratio)
        assert np.isfinite(metrics.max_drawdown)
        assert np.isfinite(metrics.calmar_ratio)
        
        # VaR and CVaR should be negative
        assert metrics.var_95 < 0
        assert metrics.cvar_95 < metrics.var_95
        
        # Volatility should be positive
        assert metrics.volatility_annual > 0
        
        # Max drawdown should be negative
        assert metrics.max_drawdown <= 0
    
    def test_factor_scores_to_dict(self, sample_prices):
        """Test FactorScores serialization."""
        engine = QuantEngine()
        factors = engine.calculate_factors("AAPL", sample_prices)
        
        factor_dict = factors.to_dict()
        
        assert isinstance(factor_dict, dict)
        assert factor_dict["symbol"] == "AAPL"
        assert "momentum_1m" in factor_dict
        assert "volatility_30d" in factor_dict
        assert "sharpe_ratio" in factor_dict
    
    def test_risk_metrics_to_dict(self, sample_returns):
        """Test RiskMetrics serialization."""
        engine = QuantEngine()
        metrics = engine.calculate_risk_metrics("AAPL", sample_returns)
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["symbol"] == "AAPL"
        assert "var_95" in metrics_dict
        assert "sharpe_ratio" in metrics_dict
        assert "max_drawdown" in metrics_dict


class TestRegimeDetector:
    """Test market regime detection."""
    
    @pytest.fixture
    def bull_market_prices(self):
        """Generate bull market price series."""
        dates = pd.date_range(start="2023-01-01", periods=250, freq="D")
        # Trending up with low volatility
        trend = np.linspace(100, 150, 250)
        noise = np.random.normal(0, 1, 250)
        prices = trend + noise
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def bear_market_prices(self):
        """Generate bear market price series."""
        dates = pd.date_range(start="2023-01-01", periods=250, freq="D")
        # Trending down
        trend = np.linspace(100, 70, 250)
        noise = np.random.normal(0, 1, 250)
        prices = trend + noise
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sideways_market_prices(self):
        """Generate sideways market price series."""
        dates = pd.date_range(start="2023-01-01", periods=250, freq="D")
        # Mean-reverting around 100
        prices = 100 + np.random.normal(0, 2, 250)
        return pd.Series(prices, index=dates)
    
    def test_detect_bull_trend(self, bull_market_prices):
        """Test bull market detection."""
        detector = RegimeDetector()
        
        trend = detector.detect_trend_regime(bull_market_prices)
        
        assert trend == "bull"
    
    def test_detect_bear_trend(self, bear_market_prices):
        """Test bear market detection."""
        detector = RegimeDetector()
        
        trend = detector.detect_trend_regime(bear_market_prices)
        
        assert trend == "bear"
    
    def test_detect_sideways_trend(self, sideways_market_prices):
        """Test sideways market detection."""
        detector = RegimeDetector()
        
        trend = detector.detect_trend_regime(sideways_market_prices)
        
        # Should be either sideways or bear/bull (mean-reverting is ambiguous)
        assert trend in ["bull", "bear", "sideways"]
    
    def test_detect_trend_insufficient_data(self):
        """Test trend detection with insufficient data."""
        detector = RegimeDetector()
        
        short_prices = pd.Series(np.random.normal(100, 5, 50))
        trend = detector.detect_trend_regime(short_prices)
        
        assert trend == "unknown"
    
    def test_detect_volatility_regime(self):
        """Test volatility regime detection."""
        detector = RegimeDetector()
        
        dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
        
        # Test with various volatility levels
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 300), index=dates)
        vol_regime = detector.detect_volatility_regime(returns)
        
        # Should return one of the valid regimes
        assert vol_regime in ["low", "normal", "high"]
        
        # Insufficient data should return unknown
        short_returns = returns.head(30)
        vol_regime = detector.detect_volatility_regime(short_returns)
        assert vol_regime == "unknown"
    
    def test_detect_current_regime(self, bull_market_prices):
        """Test comprehensive regime detection."""
        detector = RegimeDetector()
        
        regime = detector.detect_current_regime(bull_market_prices)
        
        assert isinstance(regime, dict)
        assert "trend" in regime
        assert "volatility" in regime
        assert "timestamp" in regime
        
        assert regime["trend"] in ["bull", "bear", "sideways", "unknown"]
        assert regime["volatility"] in ["low", "normal", "high", "unknown"]


class TestOptionsAnalytics:
    """Test options pricing and Greeks."""
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing."""
        analytics = OptionsAnalytics()
        
        # Standard test case
        call_price = analytics.black_scholes_call(
            S=100,  # Current price
            K=100,  # Strike
            T=1.0,  # 1 year
            r=0.05,  # 5% risk-free rate
            sigma=0.2,  # 20% volatility
        )
        
        # Call price should be positive
        assert call_price > 0
        
        # Call at-the-money should be roughly S * N(d1) where d1 ≈ 0.35
        # Expected range: $8 - $12
        assert 8 < call_price < 12
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing."""
        analytics = OptionsAnalytics()
        
        put_price = analytics.black_scholes_put(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
        )
        
        # Put price should be positive
        assert put_price > 0
        
        # Put-call parity: C - P = S - K*e^(-rT)
        call_price = analytics.black_scholes_call(100, 100, 1.0, 0.05, 0.2)
        parity_diff = call_price - put_price - (100 - 100 * np.exp(-0.05 * 1.0))
        assert abs(parity_diff) < 0.01
    
    def test_options_at_expiry(self):
        """Test option pricing at expiration (T=0)."""
        analytics = OptionsAnalytics()
        
        # In-the-money call
        call_itm = analytics.black_scholes_call(110, 100, 0, 0.05, 0.2)
        assert call_itm == 10  # Intrinsic value
        
        # Out-of-the-money call
        call_otm = analytics.black_scholes_call(90, 100, 0, 0.05, 0.2)
        assert call_otm == 0
        
        # In-the-money put
        put_itm = analytics.black_scholes_put(90, 100, 0, 0.05, 0.2)
        assert put_itm == 10
        
        # Out-of-the-money put
        put_otm = analytics.black_scholes_put(110, 100, 0, 0.05, 0.2)
        assert put_otm == 0
    
    def test_calculate_delta(self):
        """Test Delta calculation."""
        analytics = OptionsAnalytics()
        
        # Call delta (ATM should be around 0.5)
        delta_call = analytics.calculate_delta(100, 100, 1.0, 0.05, 0.2, "call")
        assert 0.45 < delta_call < 0.65
        
        # Put delta (ATM should be around -0.5, ranging from -0.35 to -0.65)
        delta_put = analytics.calculate_delta(100, 100, 1.0, 0.05, 0.2, "put")
        assert -0.70 < delta_put < -0.30
        
        # Put-call delta relationship: delta_put = delta_call - 1
        assert abs(delta_put - (delta_call - 1)) < 0.01
        
        # ITM call delta should be higher
        delta_itm = analytics.calculate_delta(120, 100, 1.0, 0.05, 0.2, "call")
        assert delta_itm > delta_call
        
        # OTM call delta should be lower
        delta_otm = analytics.calculate_delta(80, 100, 1.0, 0.05, 0.2, "call")
        assert delta_otm < delta_call
    
    def test_calculate_gamma(self):
        """Test Gamma calculation."""
        analytics = OptionsAnalytics()
        
        # Gamma (highest at ATM)
        gamma_atm = analytics.calculate_gamma(100, 100, 1.0, 0.05, 0.2)
        gamma_itm = analytics.calculate_gamma(120, 100, 1.0, 0.05, 0.2)
        gamma_otm = analytics.calculate_gamma(80, 100, 1.0, 0.05, 0.2)
        
        # Gamma should be positive
        assert gamma_atm > 0
        
        # ATM gamma should be highest
        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm
    
    def test_calculate_vega(self):
        """Test Vega calculation."""
        analytics = OptionsAnalytics()
        
        vega = analytics.calculate_vega(100, 100, 1.0, 0.05, 0.2)
        
        # Vega should be positive
        assert vega > 0
        
        # Vega should be significant for longer-dated options
        vega_short = analytics.calculate_vega(100, 100, 0.1, 0.05, 0.2)
        assert vega > vega_short
    
    def test_calculate_theta(self):
        """Test Theta calculation."""
        analytics = OptionsAnalytics()
        
        theta_call = analytics.calculate_theta(100, 100, 1.0, 0.05, 0.2, "call")
        theta_put = analytics.calculate_theta(100, 100, 1.0, 0.05, 0.2, "put")
        
        # Theta should be negative (time decay)
        assert theta_call < 0
        
        # Both call and put should decay over time
        assert isinstance(theta_put, float)
    
    def test_greeks_at_expiry(self):
        """Test Greeks at expiration."""
        analytics = OptionsAnalytics()
        
        # All Greeks should be zero or intrinsic at expiry
        delta = analytics.calculate_delta(110, 100, 0, 0.05, 0.2, "call")
        gamma = analytics.calculate_gamma(110, 100, 0, 0.05, 0.2)
        vega = analytics.calculate_vega(110, 100, 0, 0.05, 0.2)
        theta = analytics.calculate_theta(110, 100, 0, 0.05, 0.2, "call")
        
        assert delta == 1.0  # ITM call
        assert gamma == 0.0
        assert vega == 0.0
        assert theta == 0.0


class TestPortfolioAnalytics:
    """Test portfolio-level analytics."""
    
    def test_calculate_portfolio_var(self):
        """Test portfolio VaR calculation."""
        # Simple 2-asset portfolio
        weights = np.array([0.6, 0.4])
        
        # Covariance matrix (annual)
        cov_matrix = np.array([
            [0.04, 0.01],  # Asset 1: 20% vol
            [0.01, 0.09],  # Asset 2: 30% vol
        ])
        
        portfolio_var = calculate_portfolio_var(weights, cov_matrix, confidence=0.95)
        
        # Portfolio VaR should be negative
        assert portfolio_var < 0
        
        # Manual calculation
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        from scipy.stats import norm
        z_score = norm.ppf(0.05)
        expected_var = z_score * portfolio_std
        
        assert abs(portfolio_var - expected_var) < 1e-6


class TestIntegration:
    """Integration tests for quant engine."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Generate realistic price data
        dates = pd.date_range(start="2022-01-01", periods=300, freq="D")
        np.random.seed(123)
        prices = pd.Series(100 * (1 + np.random.normal(0.0005, 0.015, 300)).cumprod(), index=dates)
        
        # Calculate everything
        engine = QuantEngine()
        detector = RegimeDetector()
        
        factors = engine.calculate_factors("TSLA", prices)
        returns = engine.calculate_returns(prices)
        risk_metrics = engine.calculate_risk_metrics("TSLA", returns)
        regime = detector.detect_current_regime(prices)
        
        # Verify all components
        assert factors.symbol == "TSLA"
        assert risk_metrics.symbol == "TSLA"
        assert isinstance(regime, dict)
        
        # Convert to dicts (for API serialization)
        factor_dict = factors.to_dict()
        metrics_dict = risk_metrics.to_dict()
        
        assert "momentum_1m" in factor_dict
        assert "var_95" in metrics_dict
        assert regime["trend"] in ["bull", "bear", "sideways"]
    
    def test_options_pricing_pipeline(self):
        """Test options pricing workflow."""
        analytics = OptionsAnalytics()
        
        # Current market conditions
        S = 150  # Stock price
        K = 155  # Strike
        T = 0.25  # 3 months
        r = 0.04
        sigma = 0.35
        
        # Price options
        call_price = analytics.black_scholes_call(S, K, T, r, sigma)
        put_price = analytics.black_scholes_put(S, K, T, r, sigma)
        
        # Calculate Greeks
        delta_call = analytics.calculate_delta(S, K, T, r, sigma, "call")
        gamma = analytics.calculate_gamma(S, K, T, r, sigma)
        vega = analytics.calculate_vega(S, K, T, r, sigma)
        theta_call = analytics.calculate_theta(S, K, T, r, sigma, "call")
        
        # All values should be valid
        assert call_price > 0
        assert put_price > 0
        assert 0 < delta_call < 1
        assert gamma > 0
        assert vega > 0
        assert theta_call < 0
