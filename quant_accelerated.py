"""
High-Performance Quantitative Finance Library
Combines C++ implementations for performance-critical calculations with Python convenience
"""

try:
    import quant_cpp
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    import warnings
    error_msg = str(e)
    if "No module named 'quant_cpp'" in error_msg:
        warnings.warn(
            "C++ extensions not available. Using pure Python implementations. "
            "Build C++ extensions with: python setup_cpp.py build_ext --inplace"
        )
    else:
        warnings.warn(
            f"C++ extensions import failed: {error_msg}. "
            "Using pure Python implementations."
        )

from typing import Union, List, Optional
import numpy as np


class BlackScholesAccelerated:
    """
    Black-Scholes options pricing with C++ acceleration when available.
    Falls back to pure Python if C++ library is not built.
    """
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate European call option price.
        Uses C++ implementation for ~10x speedup when available.
        """
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.call_price(S, K, T, r, sigma, q)
        else:
            # Fallback to Python implementation
            from models.options.black_scholes import BlackScholes
            return BlackScholes.call_price(S, K, T, r, sigma, q)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate European put option price."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.put_price(S, K, T, r, sigma, q)
        else:
            from models.options.black_scholes import BlackScholes
            return BlackScholes.put_price(S, K, T, r, sigma, q)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
              is_call: bool = True, q: float = 0.0) -> float:
        """Calculate option delta."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.delta(S, K, T, r, sigma, is_call, q)
        else:
            from models.options.black_scholes import BlackScholes
            option_type = 'call' if is_call else 'put'
            return BlackScholes.delta(S, K, T, r, sigma, option_type, q)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate option gamma."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.gamma(S, K, T, r, sigma, q)
        else:
            from models.options.black_scholes import BlackScholes
            return BlackScholes.gamma(S, K, T, r, sigma, q)
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate option vega."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.vega(S, K, T, r, sigma, q)
        else:
            from models.options.black_scholes import BlackScholes
            return BlackScholes.vega(S, K, T, r, sigma, q)
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, 
              is_call: bool = True, q: float = 0.0) -> float:
        """Calculate option theta."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.theta(S, K, T, r, sigma, is_call, q)
        else:
            from models.options.black_scholes import BlackScholes
            option_type = 'call' if is_call else 'put'
            return BlackScholes.theta(S, K, T, r, sigma, option_type, q)
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, 
            is_call: bool = True, q: float = 0.0) -> float:
        """Calculate option rho."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.rho(S, K, T, r, sigma, is_call, q)
        else:
            from models.options.black_scholes import BlackScholes
            option_type = 'call' if is_call else 'put'
            return BlackScholes.rho(S, K, T, r, sigma, option_type, q)
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, r: float,
                          is_call: bool = True, q: float = 0.0) -> float:
        """Calculate implied volatility."""
        if CPP_AVAILABLE:
            return quant_cpp.BlackScholes.implied_volatility(market_price, S, K, T, r, is_call, q)
        else:
            from models.options.black_scholes import BlackScholes
            option_type = 'call' if is_call else 'put'
            return BlackScholes.implied_volatility(market_price, S, K, T, r, option_type, q)


class MonteCarloAccelerated:
    """
    Monte Carlo simulation engine with C++ acceleration.
    Provides 50-100x speedup for large simulations.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize Monte Carlo engine."""
        if CPP_AVAILABLE:
            self.engine = quant_cpp.MonteCarloEngine(seed)
        else:
            self.seed = seed
            np.random.seed(seed)
    
    def price_european_option(self, S0: float, K: float, T: float, r: float,
                             sigma: float, is_call: bool, n_simulations: int,
                             q: float = 0.0) -> float:
        """Price European option using Monte Carlo."""
        if CPP_AVAILABLE:
            return self.engine.price_european_option(S0, K, T, r, sigma, is_call, n_simulations, q)
        else:
            # Pure Python fallback
            z = np.random.standard_normal(n_simulations)
            ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
            payoff = np.maximum(ST - K, 0) if is_call else np.maximum(K - ST, 0)
            return np.exp(-r * T) * np.mean(payoff)
    
    def price_asian_option(self, S0: float, K: float, T: float, r: float,
                          sigma: float, is_call: bool, n_simulations: int,
                          n_steps: int) -> float:
        """Price Asian option using Monte Carlo."""
        if CPP_AVAILABLE:
            return self.engine.price_asian_option(S0, K, T, r, sigma, is_call, n_simulations, n_steps)
        else:
            # Pure Python fallback
            dt = T / n_steps
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            
            payoffs = []
            for _ in range(n_simulations):
                path = [S0]
                for _ in range(n_steps):
                    z = np.random.standard_normal()
                    path.append(path[-1] * np.exp(drift + diffusion * z))
                avg_price = np.mean(path)
                payoff = max(avg_price - K, 0) if is_call else max(K - avg_price, 0)
                payoffs.append(payoff)
            
            return np.exp(-r * T) * np.mean(payoffs)
    
    def simulate_gbm_path(self, S0: float, mu: float, sigma: float, T: float, steps: int) -> List[float]:
        """Simulate geometric Brownian motion path."""
        if CPP_AVAILABLE:
            return self.engine.simulate_gbm_path(S0, mu, sigma, T, steps)
        else:
            dt = T / steps
            path = [S0]
            for _ in range(steps):
                z = np.random.standard_normal()
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z))
            return path


class PortfolioAccelerated:
    """
    Portfolio analytics with C++ acceleration.
    Provides significant speedup for large-scale portfolio calculations.
    """
    
    @staticmethod
    def portfolio_return(weights: Union[List[float], np.ndarray], 
                        expected_returns: Union[List[float], np.ndarray]) -> float:
        """Calculate portfolio expected return."""
        if CPP_AVAILABLE and isinstance(weights, list) and isinstance(expected_returns, list):
            return quant_cpp.Portfolio.portfolio_return(weights, expected_returns)
        else:
            weights = np.array(weights)
            expected_returns = np.array(expected_returns)
            return float(np.dot(weights, expected_returns))
    
    @staticmethod
    def portfolio_volatility(weights: Union[List[float], np.ndarray],
                            cov_matrix: Union[List[List[float]], np.ndarray]) -> float:
        """Calculate portfolio volatility."""
        if CPP_AVAILABLE and isinstance(weights, list):
            cov_list = cov_matrix.tolist() if isinstance(cov_matrix, np.ndarray) else cov_matrix
            return quant_cpp.Portfolio.portfolio_volatility(weights, cov_list)
        else:
            weights = np.array(weights)
            cov_matrix = np.array(cov_matrix)
            return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    
    @staticmethod
    def sharpe_ratio(weights: Union[List[float], np.ndarray],
                    expected_returns: Union[List[float], np.ndarray],
                    cov_matrix: Union[List[List[float]], np.ndarray],
                    risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if CPP_AVAILABLE and isinstance(weights, list):
            cov_list = cov_matrix.tolist() if isinstance(cov_matrix, np.ndarray) else cov_matrix
            exp_ret_list = expected_returns.tolist() if isinstance(expected_returns, np.ndarray) else expected_returns
            return quant_cpp.Portfolio.sharpe_ratio(weights, exp_ret_list, cov_list, risk_free_rate)
        else:
            ret = PortfolioAccelerated.portfolio_return(weights, expected_returns)
            vol = PortfolioAccelerated.portfolio_volatility(weights, cov_matrix)
            return (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
    
    @staticmethod
    def max_drawdown(cumulative_returns: Union[List[float], np.ndarray]) -> float:
        """Calculate maximum drawdown."""
        if CPP_AVAILABLE and isinstance(cumulative_returns, list):
            return quant_cpp.Portfolio.max_drawdown(cumulative_returns)
        else:
            cumulative_returns = np.array(cumulative_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            return float(np.max(drawdown))
    
    @staticmethod
    def historical_var(returns: Union[List[float], np.ndarray], confidence_level: float) -> float:
        """Calculate Value at Risk (historical method)."""
        if CPP_AVAILABLE and isinstance(returns, list):
            return quant_cpp.Portfolio.historical_var(returns, confidence_level)
        else:
            returns = np.array(returns)
            sorted_returns = np.sort(returns)
            index = int((1.0 - confidence_level) * len(sorted_returns))
            return -float(sorted_returns[index])
    
    @staticmethod
    def conditional_var(returns: Union[List[float], np.ndarray], confidence_level: float) -> float:
        """Calculate Conditional Value at Risk."""
        if CPP_AVAILABLE and isinstance(returns, list):
            return quant_cpp.Portfolio.conditional_var(returns, confidence_level)
        else:
            returns = np.array(returns)
            sorted_returns = np.sort(returns)
            index = int((1.0 - confidence_level) * len(sorted_returns))
            return -float(np.mean(sorted_returns[:index+1]))


# Export main classes
__all__ = [
    'BlackScholesAccelerated',
    'MonteCarloAccelerated', 
    'PortfolioAccelerated',
    'CPP_AVAILABLE'
]
