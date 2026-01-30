"""
Advanced Options Pricing Models
Heston, SABR, Binomial Tree, Finite Difference Methods
"""

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from models.options.black_scholes import BlackScholes


class BinomialTree:
    """
    Binomial Tree Options Pricing Model.
    More flexible than Black-Scholes, handles American options.
    """
    
    def __init__(self, n_steps: int = 100):
        """
        Initialize binomial tree.
        
        Args:
            n_steps: Number of time steps
        """
        self.n_steps = n_steps
    
    def call_price(self,
                   S: float,
                   K: float,
                   T: float,
                   r: float,
                   sigma: float,
                   dividend_yield: float = 0.0,
                   american: bool = False) -> float:
        """
        Price European or American call option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Dividend yield
            american: American option (early exercise)
        
        Returns:
            Option price
        """
        dt = T / self.n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)
        
        # Stock prices at expiration
        stock_prices = np.zeros(self.n_steps + 1)
        for i in range(self.n_steps + 1):
            stock_prices[i] = S * (u ** (self.n_steps - i)) * (d ** i)
        
        # Option values at expiration
        option_values = np.maximum(stock_prices - K, 0)
        
        # Backward induction
        for step in range(self.n_steps - 1, -1, -1):
            for i in range(step + 1):
                stock_price = S * (u ** (step - i)) * (d ** i)
                
                # Expected value
                expected_value = (p * option_values[i] + 
                                (1 - p) * option_values[i + 1]) * np.exp(-r * dt)
                
                if american:
                    # Check early exercise
                    intrinsic = max(stock_price - K, 0)
                    option_values[i] = max(expected_value, intrinsic)
                else:
                    option_values[i] = expected_value
        
        return option_values[0]
    
    def put_price(self,
                  S: float,
                  K: float,
                  T: float,
                  r: float,
                  sigma: float,
                  dividend_yield: float = 0.0,
                  american: bool = False) -> float:
        """
        Price European or American put option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Dividend yield
            american: American option
        
        Returns:
            Option price
        """
        # Use put-call parity for European
        if not american:
            call = self.call_price(S, K, T, r, sigma, dividend_yield, False)
            put = call - S * np.exp(-dividend_yield * T) + K * np.exp(-r * T)
            return put
        
        # American put requires separate calculation
        dt = T / self.n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)
        
        stock_prices = np.zeros(self.n_steps + 1)
        for i in range(self.n_steps + 1):
            stock_prices[i] = S * (u ** (self.n_steps - i)) * (d ** i)
        
        option_values = np.maximum(K - stock_prices, 0)
        
        for step in range(self.n_steps - 1, -1, -1):
            for i in range(step + 1):
                stock_price = S * (u ** (step - i)) * (d ** i)
                expected_value = (p * option_values[i] + 
                                (1 - p) * option_values[i + 1]) * np.exp(-r * dt)
                intrinsic = max(K - stock_price, 0)
                option_values[i] = max(expected_value, intrinsic)
        
        return option_values[0]


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Model for Volatility Smile.
    Used by institutional traders for interest rate derivatives.
    """
    
    def __init__(self):
        """Initialize SABR model."""
        self.alpha = None
        self.beta = None
        self.rho = None
        self.nu = None
    
    def implied_volatility(self,
                          F: float,
                          K: float,
                          T: float,
                          alpha: float,
                          beta: float,
                          rho: float,
                          nu: float) -> float:
        """
        Calculate SABR implied volatility.
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiration
            alpha: Volatility parameter
            beta: Skew parameter
            rho: Correlation
            nu: Volatility of volatility
        
        Returns:
            Implied volatility
        """
        if F == K:
            # At-the-money formula
            return alpha / (F ** (1 - beta))
        
        z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        
        numerator = alpha * (1 + 
            ((1 - beta) ** 2 / 24) * (np.log(F / K) ** 2) +
            ((1 - beta) ** 4 / 1920) * (np.log(F / K) ** 4))
        
        denominator = (F * K) ** ((1 - beta) / 2) * (1 + 
            ((1 - beta) ** 2 / 24) * (np.log(F / K) ** 2) +
            ((1 - beta) ** 4 / 1920) * (np.log(F / K) ** 4)) * x
        
        if abs(denominator) < 1e-10:
            return alpha / (F ** (1 - beta))
        
        iv = numerator / denominator
        
        # Correction term
        correction = ((1 - beta) ** 2 / 24) * (alpha ** 2) / ((F * K) ** (1 - beta))
        correction += (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
        correction += ((2 - 3 * rho ** 2) / 24) * nu ** 2
        
        iv *= (1 + correction * T)
        
        return iv
    
    def calibrate(self,
                 strikes: np.ndarray,
                 market_vols: np.ndarray,
                 F: float,
                 T: float,
                 beta: float = 0.5) -> Dict[str, float]:
        """
        Calibrate SABR parameters to market volatilities.
        
        Args:
            strikes: Strike prices
            market_vols: Market implied volatilities
            F: Forward price
            T: Time to expiration
            beta: Fixed beta (typically 0.5 for equity)
        
        Returns:
            Calibrated parameters
        """
        def objective(params):
            alpha, rho, nu = params
            errors = []
            for K, market_vol in zip(strikes, market_vols):
                model_vol = self.implied_volatility(F, K, T, alpha, beta, rho, nu)
                errors.append((model_vol - market_vol) ** 2)
            return np.sum(errors)
        
        # Initial guess
        initial = [0.2, -0.5, 0.5]
        bounds = [(0.01, 1.0), (-0.99, 0.99), (0.01, 2.0)]
        
        result = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)
        
        self.alpha = result.x[0]
        self.beta = beta
        self.rho = result.x[1]
        self.nu = result.x[2]
        
        return {
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'rho': float(self.rho),
            'nu': float(self.nu)
        }


class FiniteDifferencePricing:
    """
    Finite Difference Method for Options Pricing.
    Solves Black-Scholes PDE numerically.
    """
    
    def __init__(self, n_steps_S: int = 100, n_steps_t: int = 100):
        """
        Initialize finite difference solver.
        
        Args:
            n_steps_S: Number of price steps
            n_steps_t: Number of time steps
        """
        self.n_steps_S = n_steps_S
        self.n_steps_t = n_steps_t
    
    def call_price(self,
                   S: float,
                   K: float,
                   T: float,
                   r: float,
                   sigma: float,
                   S_max: float = None) -> float:
        """
        Price European call using finite difference.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            S_max: Maximum stock price for grid
        
        Returns:
            Option price
        """
        if S_max is None:
            S_max = S * 3
        
        dS = S_max / self.n_steps_S
        dt = T / self.n_steps_t
        
        # Grid
        S_grid = np.linspace(0, S_max, self.n_steps_S + 1)
        t_grid = np.linspace(0, T, self.n_steps_t + 1)
        
        # Option values
        V = np.zeros((self.n_steps_t + 1, self.n_steps_S + 1))
        
        # Boundary conditions at expiration
        V[-1, :] = np.maximum(S_grid - K, 0)
        
        # Boundary conditions
        V[:, 0] = 0  # S = 0
        V[:, -1] = S_max - K * np.exp(-r * (T - t_grid))  # S = S_max
        
        # Backward induction (Crank-Nicolson)
        for i in range(self.n_steps_t - 1, -1, -1):
            for j in range(1, self.n_steps_S):
                S_j = S_grid[j]
                
                # Finite difference coefficients
                alpha = 0.25 * dt * (sigma ** 2 * j ** 2 - r * j)
                beta = -0.5 * dt * (sigma ** 2 * j ** 2 + r)
                gamma = 0.25 * dt * (sigma ** 2 * j ** 2 + r * j)
                
                V[i, j] = (V[i+1, j] + 
                           alpha * V[i+1, j-1] + 
                           beta * V[i+1, j] + 
                           gamma * V[i+1, j+1])
        
        # Interpolate to current price
        from scipy.interpolate import interp1d
        f = interp1d(S_grid, V[0, :], kind='linear')
        return float(f(S))
