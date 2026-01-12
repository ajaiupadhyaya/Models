"""
Utility functions for financial modeling.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime, timedelta


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
    
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    if downside_std == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Returns series
    
    Returns:
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Returns series
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
        VaR value
    """
    return returns.quantile(confidence_level)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional VaR (CVaR) / Expected Shortfall.
    
    Args:
        returns: Returns series
        confidence_level: Confidence level
    
    Returns:
        CVaR value
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize returns.
    
    Args:
        returns: Returns series
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized return
    """
    return (1 + returns.mean()) ** periods_per_year - 1


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize volatility.
    
    Args:
        returns: Returns series
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta of asset relative to market.
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
    
    Returns:
        Beta coefficient
    """
    covariance = np.cov(asset_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix.
    
    Args:
        returns_df: DataFrame with returns
    
    Returns:
        Correlation matrix
    """
    return returns_df.corr()


def calculate_rolling_statistics(data: pd.Series, 
                                 window: int = 30,
                                 stat: str = 'mean') -> pd.Series:
    """
    Calculate rolling statistics.
    
    Args:
        data: Data series
        window: Rolling window size
        stat: Statistic ('mean', 'std', 'min', 'max')
    
    Returns:
        Rolling statistic series
    """
    if stat == 'mean':
        return data.rolling(window=window).mean()
    elif stat == 'std':
        return data.rolling(window=window).std()
    elif stat == 'min':
        return data.rolling(window=window).min()
    elif stat == 'max':
        return data.rolling(window=window).max()
    else:
        raise ValueError(f"Unsupported stat: {stat}")


def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format number as currency.
    
    Args:
        value: Numeric value
        currency: Currency code
    
    Returns:
        Formatted string
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Numeric value
        decimals: Decimal places
    
    Returns:
        Formatted string
    """
    return f"{value * 100:.{decimals}f}%"
