"""
Value at Risk (VaR) and Conditional VaR (CVaR) Models
Multiple methodologies: Historical, Parametric, Monte Carlo
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class VaRModel:
    """
    Value at Risk calculation using multiple methodologies.
    """
    
    @staticmethod
    def historical_var(returns: pd.Series, 
                      confidence_level: float = 0.05) -> float:
        """
        Calculate VaR using historical simulation.
        
        Args:
            returns: Returns series
            confidence_level: Confidence level (0.05 for 95% VaR)
        
        Returns:
            VaR value
        """
        return returns.quantile(confidence_level)
    
    @staticmethod
    def parametric_var(returns: pd.Series,
                      confidence_level: float = 0.05,
                      assume_normal: bool = True) -> float:
        """
        Calculate VaR using parametric method.
        
        Args:
            returns: Returns series
            confidence_level: Confidence level
            assume_normal: Assume normal distribution
        
        Returns:
            VaR value
        """
        mean = returns.mean()
        std = returns.std()
        
        if assume_normal:
            z_score = stats.norm.ppf(confidence_level)
        else:
            # Use t-distribution
            z_score = stats.t.ppf(confidence_level, len(returns) - 1)
        
        var = mean + z_score * std
        return var
    
    @staticmethod
    def monte_carlo_var(returns: pd.Series,
                       confidence_level: float = 0.05,
                       n_simulations: int = 10000,
                       time_horizon: int = 1) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            returns: Returns series
            confidence_level: Confidence level
            n_simulations: Number of simulations
            time_horizon: Time horizon (days)
        
        Returns:
            VaR value
        """
        mean = returns.mean()
        std = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean, std, (n_simulations, time_horizon))
        portfolio_returns = simulated_returns.sum(axis=1)
        
        var = np.percentile(portfolio_returns, confidence_level * 100)
        return var
    
    @staticmethod
    def calculate_var(returns: pd.Series,
                     method: str = 'historical',
                     confidence_level: float = 0.05,
                     **kwargs) -> float:
        """
        Calculate VaR using specified method.
        
        Args:
            returns: Returns series
            method: 'historical', 'parametric', or 'monte_carlo'
            confidence_level: Confidence level
            **kwargs: Additional method-specific parameters
        
        Returns:
            VaR value
        """
        if method == 'historical':
            return VaRModel.historical_var(returns, confidence_level)
        elif method == 'parametric':
            return VaRModel.parametric_var(returns, confidence_level, 
                                          kwargs.get('assume_normal', True))
        elif method == 'monte_carlo':
            return VaRModel.monte_carlo_var(returns, confidence_level,
                                          kwargs.get('n_simulations', 10000),
                                          kwargs.get('time_horizon', 1))
        else:
            raise ValueError(f"Unknown method: {method}")


class CVaRModel:
    """
    Conditional VaR (Expected Shortfall) calculation.
    """
    
    @staticmethod
    def historical_cvar(returns: pd.Series,
                       confidence_level: float = 0.05) -> float:
        """
        Calculate CVaR using historical simulation.
        
        Args:
            returns: Returns series
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        var = VaRModel.historical_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def parametric_cvar(returns: pd.Series,
                       confidence_level: float = 0.05) -> float:
        """
        Calculate CVaR using parametric method.
        
        Args:
            returns: Returns series
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(confidence_level)
        
        # CVaR formula for normal distribution
        cvar = mean - std * (stats.norm.pdf(z_score) / confidence_level)
        return cvar
    
    @staticmethod
    def calculate_cvar(returns: pd.Series,
                      method: str = 'historical',
                      confidence_level: float = 0.05) -> float:
        """
        Calculate CVaR using specified method.
        
        Args:
            returns: Returns series
            method: 'historical' or 'parametric'
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        if method == 'historical':
            return CVaRModel.historical_cvar(returns, confidence_level)
        elif method == 'parametric':
            return CVaRModel.parametric_cvar(returns, confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")


class StressTest:
    """
    Portfolio stress testing framework.
    """
    
    @staticmethod
    def scenario_analysis(returns: pd.Series,
                         scenarios: List[float]) -> pd.DataFrame:
        """
        Perform scenario analysis.
        
        Args:
            returns: Returns series
            scenarios: List of scenario return shocks (e.g., [-0.1, -0.2, -0.3])
        
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        for scenario in scenarios:
            stressed_returns = returns + scenario
            var = VaRModel.historical_var(stressed_returns)
            cvar = CVaRModel.historical_cvar(stressed_returns)
            
            results.append({
                'scenario_shock': scenario,
                'var': var,
                'cvar': cvar,
                'mean_return': stressed_returns.mean(),
                'volatility': stressed_returns.std()
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def historical_stress(returns: pd.Series,
                         stress_periods: List[tuple]) -> pd.DataFrame:
        """
        Stress test using historical periods.
        
        Args:
            returns: Returns series
            stress_periods: List of (start_date, end_date) tuples
        
        Returns:
            DataFrame with stress period results
        """
        results = []
        
        for start, end in stress_periods:
            period_returns = returns.loc[start:end]
            var = VaRModel.historical_var(period_returns)
            cvar = CVaRModel.historical_cvar(period_returns)
            
            results.append({
                'period': f"{start} to {end}",
                'var': var,
                'cvar': cvar,
                'mean_return': period_returns.mean(),
                'volatility': period_returns.std(),
                'min_return': period_returns.min(),
                'max_drawdown': (1 + period_returns).cumprod().pct_change().min()
            })
        
        return pd.DataFrame(results)
