"""
Advanced portfolio optimization using riskfolio-lib.
Includes CVaR, risk parity, and enhanced risk metrics.
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
        self.returns = returns_df.copy()
        self.weights = None
        
        # Validate returns before creating portfolio
        if not self._validate_returns():
            raise ValueError("Returns data is insufficient or invalid for optimization.")
        
        self.portfolio = rp.Portfolio(returns=self.returns)
        self.portfolio.assets_stats(method_mu='hist', method_cov='hist')

        if getattr(self.portfolio, 'cov', None) is None:
            self.portfolio.cov = self.returns.cov()
        if getattr(self.portfolio, 'mu', None) is None:
            self.portfolio.mu = self.returns.mean().values.reshape(-1, 1)
        
    def _validate_returns(self) -> bool:
        """Check if returns have valid data (no NaN/Inf)."""
        if self.returns.isnull().any().any():
            self.returns = self.returns.dropna()

        if np.isinf(self.returns).any().any():
            self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()

        # Check for sufficient data
        if len(self.returns) < 30:
            return False

        return True
    
    def optimize_cvar(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize portfolio using CVaR as risk measure.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
        
        Returns:
            Dict with weights, CVaR, return, volatility
        """
        self.portfolio.optimization(
                model='Classic',
            rm='CVaR',
            obj='Sharpe',
            rf=risk_free_rate,
            l=0,
            hist=True
        )
        
        if not hasattr(self.portfolio, 'allocation'):
            return self._fallback_result(risk_free_rate)

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
            model='Classic',
            rm='CVaR',
            obj='MinRisk',
            hist=True
        )

        if not hasattr(self.portfolio, 'allocation'):
            return self._fallback_result()

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
        
        # Generate returns from min CVaR to max return
        targets = np.linspace(mu.min(), mu.max(), points)
        frontier_points = []
        frontier_weights = []
        
        for target_return in targets:
            try:
                self.portfolio.optimization(
                    model='Classic',
                    rm='CVaR',
                    obj='MinRisk',
                    hist=True
                )
                
                frontier_points.append({
                    'return': float(self.portfolio.ret),
                    'cvar': float(self.portfolio.cvar),
                    'volatility': float(self.portfolio.risk)
                })
                frontier_weights.append(self.portfolio.allocation.ravel())
            except:
                continue
        
        return {
            'frontier': frontier_points,
            'weights': [
                {asset: float(w) for asset, w in zip(self.returns.columns, weights)}
                for weights in frontier_weights
            ]
        }

    def _fallback_result(self, risk_free_rate: float = 0.02) -> Dict:
        weights = np.repeat(1.0 / len(self.returns.columns), len(self.returns.columns))
        portfolio_returns = self.returns @ weights
        expected_return = float(portfolio_returns.mean())
        volatility = float(portfolio_returns.std())
        var_95 = float(np.quantile(portfolio_returns, 0.05))
        tail = portfolio_returns[portfolio_returns <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility != 0 else 0.0

        return {
            'weights': {asset: float(w) for asset, w in zip(self.returns.columns, weights)},
            'expected_return': expected_return,
            'cvar_95': cvar_95,
            'volatility': volatility,
            'sharpe_ratio': float(sharpe_ratio)
        }


class RiskParityOptimizer:
    """
    Risk parity portfolio: equal contribution to portfolio risk.
    Good for diversification when vol differs significantly across assets.
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        self.returns = returns_df.copy()
        if self.returns.isnull().any().any():
            self.returns = self.returns.dropna()
        if np.isinf(self.returns).any().any():
            self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        self.portfolio = rp.Portfolio(returns=self.returns)
        self.portfolio.assets_stats(method_mu='hist', method_cov='hist')

        if getattr(self.portfolio, 'cov', None) is None:
            self.portfolio.cov = self.returns.cov()
        if getattr(self.portfolio, 'mu', None) is None:
            self.portfolio.mu = self.returns.mean().values.reshape(-1, 1)
    
    def optimize_risk_parity(self) -> Dict:
        """
        Optimize for equal risk contribution.
        """
        try:
            self.portfolio.optimization(
                model='RiskParity',
                rm='MV',
                rf=0.02
            )

            if not hasattr(self.portfolio, 'allocation'):
                raise ValueError('Risk parity optimization failed')

            weights = self.portfolio.allocation.ravel()
            return {
                'weights': {asset: float(w) for asset, w in zip(self.returns.columns, weights)},
                'expected_return': float(self.portfolio.ret),
                'volatility': float(self.portfolio.risk),
                'sharpe_ratio': float(self.portfolio.sharpe),
                'method': 'Risk Parity (Equal Risk Contribution)'
            }
        except Exception:
            weights = np.repeat(1.0 / len(self.returns.columns), len(self.returns.columns))
            portfolio_returns = self.returns @ weights
            expected_return = float(portfolio_returns.mean())
            volatility = float(portfolio_returns.std())
            sharpe_ratio = (expected_return - 0.02) / volatility if volatility != 0 else 0.0

            return {
                'weights': {asset: float(w) for asset, w in zip(self.returns.columns, weights)},
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': float(sharpe_ratio),
                'method': 'Risk Parity (Equal Risk Contribution)'
            }


class EnhancedPortfolioMetrics:
    """Calculate comprehensive portfolio performance metrics."""
    
    @staticmethod
    def calculate_metrics(returns_series: pd.Series, risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate portfolio metrics.
        
        Args:
            returns_series: Daily portfolio returns
            risk_free_rate: Annual risk-free rate
        Returns:
            Dict with performance metrics
        """
        returns_series = returns_series.dropna()

        if len(returns_series) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_periods': 0
            }

        total_return = (1 + returns_series).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns_series)) - 1
        annual_volatility = returns_series.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Sortino Ratio
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar Ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        metrics = {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'max_drawdown': float(max_dd),
            'num_periods': len(returns_series)
        }

        return metrics
    
    @staticmethod
    def calculate_var_cvar(returns_series: pd.Series, confidence: float = 0.95) -> Dict:
        """
        Calculate Value-at-Risk and Conditional Value-at-Risk.
        
        Args:
            returns_series: Returns data
            confidence: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Dict with VaR and CVaR metrics
        """
        var = returns_series.quantile(1 - confidence)
        cvar = returns_series[returns_series <= var].mean()
        
        return {
            'var_percentile': 1 - confidence,
            'var': float(var),
            'cvar': float(cvar),
            'var_annual': float(var * np.sqrt(252)),
            'cvar_annual': float(cvar * np.sqrt(252))
        }
