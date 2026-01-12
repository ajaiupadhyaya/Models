"""
Portfolio Optimization Models
Mean-Variance Optimization, Risk Parity, Black-Litterman
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimization (Markowitz).
    """
    
    def __init__(self, 
                 expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize optimizer.
        
        Args:
            expected_returns: Expected returns for assets
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def optimize_sharpe(self) -> Dict:
        """
        Optimize for maximum Sharpe ratio.
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'weights': pd.Series(optimal_weights, index=self.expected_returns.index),
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe
        }
    
    def optimize_min_volatility(self) -> Dict:
        """
        Optimize for minimum volatility.
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        
        return {
            'weights': pd.Series(optimal_weights, index=self.expected_returns.index),
            'expected_return': portfolio_return,
            'volatility': portfolio_std
        }
    
    def optimize_target_return(self, target_return: float) -> Dict:
        """
        Optimize for target return with minimum volatility.
        
        Args:
            target_return: Target portfolio return
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, self.expected_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        
        return {
            'weights': pd.Series(optimal_weights, index=self.expected_returns.index),
            'expected_return': portfolio_return,
            'volatility': portfolio_std
        }


class RiskParityOptimizer:
    """
    Risk Parity Portfolio Optimization.
    """
    
    def __init__(self, cov_matrix: pd.DataFrame):
        """
        Initialize risk parity optimizer.
        
        Args:
            cov_matrix: Covariance matrix
        """
        self.cov_matrix = cov_matrix
        self.n_assets = len(cov_matrix)
    
    def optimize(self) -> Dict:
        """
        Optimize for equal risk contribution.
        
        Returns:
            Dictionary with optimal weights
        """
        def risk_contributions(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return contrib
        
        def objective(weights):
            contrib = risk_contributions(weights)
            # Minimize sum of squared differences from equal contribution
            target_contrib = np.ones(self.n_assets) / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        
        return {
            'weights': pd.Series(optimal_weights, index=self.cov_matrix.index),
            'volatility': portfolio_vol
        }


def optimize_portfolio_from_returns(returns_df: pd.DataFrame,
                                    method: str = 'sharpe',
                                    risk_free_rate: float = 0.02) -> Dict:
    """
    Optimize portfolio from historical returns.
    
    Args:
        returns_df: DataFrame with asset returns
        method: Optimization method ('sharpe', 'min_vol', 'risk_parity')
        risk_free_rate: Risk-free rate
    
    Returns:
        Dictionary with optimization results
    """
    expected_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    if method == 'sharpe':
        optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix, risk_free_rate)
        return optimizer.optimize_sharpe()
    elif method == 'min_vol':
        optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix, risk_free_rate)
        return optimizer.optimize_min_volatility()
    elif method == 'risk_parity':
        optimizer = RiskParityOptimizer(cov_matrix)
        return optimizer.optimize()
    else:
        raise ValueError(f"Unknown method: {method}")
