"""
Quantitative Engine

Provides advanced quantitative analytics for trading and portfolio management:
- Factor models (momentum, value, quality, volatility)
- Regime detection (bull/bear/sideways markets)
- Risk metrics (VaR, CVaR, volatility)
- Portfolio optimization
- Options Greeks and pricing (Black-Scholes)

Usage:
    from core.quant_engine import QuantEngine, RegimeDetector
    
    engine = QuantEngine()
    factors = engine.calculate_factors(prices_df)
    regime = RegimeDetector().detect_current_regime(returns)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class FactorScores:
    """Factor scores for a symbol."""
    
    symbol: str
    date: datetime
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float
    volatility_30d: float
    volatility_90d: float
    sharpe_ratio: float
    max_drawdown: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "momentum_1m": self.momentum_1m,
            "momentum_3m": self.momentum_3m,
            "momentum_6m": self.momentum_6m,
            "momentum_12m": self.momentum_12m,
            "volatility_30d": self.volatility_30d,
            "volatility_90d": self.volatility_90d,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio or symbol."""
    
    symbol: str
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR (expected shortfall)
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "volatility_annual": self.volatility_annual,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
        }


class QuantEngine:
    """
    Core quantitative engine for factor calculation and risk analysis.
    
    Provides:
    - Momentum factors (1M, 3M, 6M, 12M)
    - Volatility metrics (30D, 90D)
    - Risk-adjusted returns (Sharpe, Sortino)
    - Drawdown analysis
    - VaR/CVaR calculations
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate simple returns from prices."""
        return prices.pct_change().fillna(0)
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns from prices."""
        return np.log(prices / prices.shift(1)).fillna(0)
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, periods: int) -> float:
        """
        Calculate momentum over N periods (total return).
        
        Args:
            prices: Price series
            periods: Lookback period
            
        Returns:
            Total return over period
        """
        if len(prices) < periods + 1:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[-periods - 1]) - 1.0
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annual_factor: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Return series
            annual_factor: Trading days per year (default: 252)
            
        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(annual_factor)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        annual_factor: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate (default: 2%)
            annual_factor: Trading days per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / annual_factor)
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(annual_factor) * (excess_returns.mean() / excess_returns.std())
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        annual_factor: int = 252,
    ) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility).
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            annual_factor: Trading days per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / annual_factor)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        return np.sqrt(annual_factor) * (excess_returns.mean() / downside_std)
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(prices) < 2:
            return 0.0
        
        cumulative = (1 + QuantEngine.calculate_returns(prices)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical method.
        
        Args:
            returns: Return series
            confidence: Confidence level (default: 95%)
            
        Returns:
            VaR (negative value indicates loss)
        """
        if len(returns) < 10:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Return series
            confidence: Confidence level
            
        Returns:
            CVaR (expected loss beyond VaR)
        """
        if len(returns) < 10:
            return 0.0
        
        var = QuantEngine.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_factors(
        self,
        symbol: str,
        prices: pd.Series,
    ) -> FactorScores:
        """
        Calculate all factor scores for a symbol.
        
        Args:
            symbol: Symbol identifier
            prices: Price series (indexed by date)
            
        Returns:
            FactorScores object
        """
        returns = self.calculate_returns(prices)
        
        return FactorScores(
            symbol=symbol,
            date=prices.index[-1],
            momentum_1m=self.calculate_momentum(prices, 21),
            momentum_3m=self.calculate_momentum(prices, 63),
            momentum_6m=self.calculate_momentum(prices, 126),
            momentum_12m=self.calculate_momentum(prices, 252),
            volatility_30d=self.calculate_volatility(returns.tail(30)),
            volatility_90d=self.calculate_volatility(returns.tail(90)),
            sharpe_ratio=self.calculate_sharpe_ratio(returns.tail(252)),
            max_drawdown=self.calculate_max_drawdown(prices),
        )
    
    def calculate_risk_metrics(
        self,
        symbol: str,
        returns: pd.Series,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            symbol: Symbol identifier
            returns: Return series
            
        Returns:
            RiskMetrics object
        """
        max_dd = self.calculate_max_drawdown((1 + returns).cumprod())
        sharpe = self.calculate_sharpe_ratio(returns)
        
        # Calmar ratio (return / max drawdown)
        annual_return = returns.mean() * 252
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        return RiskMetrics(
            symbol=symbol,
            var_95=self.calculate_var(returns, 0.95),
            cvar_95=self.calculate_cvar(returns, 0.95),
            volatility_annual=self.calculate_volatility(returns),
            sharpe_ratio=sharpe,
            sortino_ratio=self.calculate_sortino_ratio(returns),
            max_drawdown=max_dd,
            calmar_ratio=calmar,
        )


class RegimeDetector:
    """
    Market regime detection (bull/bear/sideways).
    
    Uses multiple indicators:
    - Trend (50-day vs 200-day MA)
    - Volatility (VIX-like measure)
    - Momentum
    """
    
    @staticmethod
    def detect_trend_regime(prices: pd.Series) -> str:
        """
        Detect trend regime using moving averages.
        
        Returns:
            "bull", "bear", or "sideways"
        """
        if len(prices) < 200:
            return "unknown"
        
        ma_50 = prices.rolling(50).mean().iloc[-1]
        ma_200 = prices.rolling(200).mean().iloc[-1]
        current = prices.iloc[-1]
        
        # Bull: price above both MAs, 50 MA > 200 MA
        if current > ma_50 and current > ma_200 and ma_50 > ma_200:
            return "bull"
        
        # Bear: price below both MAs, 50 MA < 200 MA
        if current < ma_50 and current < ma_200 and ma_50 < ma_200:
            return "bear"
        
        return "sideways"
    
    @staticmethod
    def detect_volatility_regime(returns: pd.Series) -> str:
        """
        Detect volatility regime.
        
        Returns:
            "low", "normal", or "high"
        """
        if len(returns) < 60:
            return "unknown"
        
        recent_vol = returns.tail(30).std() * np.sqrt(252)
        historical_vol = returns.tail(252).std() * np.sqrt(252)
        
        if recent_vol < historical_vol * 0.75:
            return "low"
        elif recent_vol > historical_vol * 1.25:
            return "high"
        return "normal"
    
    def detect_current_regime(
        self,
        prices: pd.Series,
    ) -> Dict[str, str]:
        """
        Detect current market regime.
        
        Args:
            prices: Price series
            
        Returns:
            Dict with trend and volatility regimes
        """
        returns = QuantEngine.calculate_returns(prices)
        
        return {
            "trend": self.detect_trend_regime(prices),
            "volatility": self.detect_volatility_regime(returns),
            "timestamp": datetime.now().isoformat(),
        }


class OptionsAnalytics:
    """
    Options pricing and Greeks (Black-Scholes model).
    
    Provides:
    - Call/Put pricing
    - Greeks (Delta, Gamma, Theta, Vega, Rho)
    """
    
    @staticmethod
    def black_scholes_call(
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility (annual)
    ) -> float:
        """Calculate Black-Scholes call option price."""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate Black-Scholes put option price."""
        if T <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def calculate_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate option Delta (rate of change w.r.t. underlying price)."""
        if T <= 0:
            return 1.0 if option_type == "call" and S > K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def calculate_gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate option Gamma (rate of change of Delta)."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def calculate_vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate option Vega (sensitivity to volatility)."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change
    
    @staticmethod
    def calculate_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate option Theta (time decay)."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365  # Daily theta
        else:
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365  # Daily theta
        
        return theta


# Convenience functions
def calculate_portfolio_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate portfolio VaR using variance-covariance method.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix of returns
        confidence: Confidence level
        
    Returns:
        Portfolio VaR
    """
    portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
    z_score = norm.ppf(1 - confidence)
    return z_score * portfolio_std
