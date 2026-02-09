"""
Options Pricing Models - Black-Scholes, Greeks, Implied Volatility
Phase 3 - Awesome Quant Integration
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from typing import Dict, Optional, Literal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BlackScholes:
    """
    Black-Scholes option pricing model with Greeks calculations.
    
    Assumptions:
    - European options only
    - No dividends (or continuous dividend yield via q parameter)
    - Constant volatility
    - Log-normal price distribution
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate d1 parameter for Black-Scholes.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annual)
            q: Continuous dividend yield
        
        Returns:
            d1 value
        """
        if T <= 0:
            raise ValueError("Time to expiration must be positive")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate d2 parameter for Black-Scholes."""
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Price a European call option using Black-Scholes.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
            q: Continuous dividend yield
        
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholes.d2(S, K, T, r, sigma, q)
        
        call = S * np.exp(-q * T) * stats.norm.cdf(d1_val) - K * np.exp(-r * T) * stats.norm.cdf(d2_val)
        return call
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Price a European put option using Black-Scholes.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
            q: Continuous dividend yield
        
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholes.d2(S, K, T, r, sigma, q)
        
        put = K * np.exp(-r * T) * stats.norm.cdf(-d2_val) - S * np.exp(-q * T) * stats.norm.cdf(-d1_val)
        return put


class GreeksCalculator:
    """
    Calculate option Greeks (sensitivities).
    
    Greeks measure how option price changes with underlying parameters:
    - Delta: Change in price w.r.t. underlying price
    - Gamma: Change in delta w.r.t. underlying price
    - Vega: Change in price w.r.t. volatility
    - Theta: Change in price w.r.t. time
    - Rho: Change in price w.r.t. risk-free rate
    """
    
    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate delta for call option.
        
        Delta ranges from 0 to 1 for calls.
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * stats.norm.cdf(d1_val)
    
    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate delta for put option.
        
        Delta ranges from -1 to 0 for puts.
        """
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        return -np.exp(-q * T) * stats.norm.cdf(-d1_val)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate gamma (same for call and put).
        
        Gamma measures rate of change of delta.
        Highest gamma near ATM, decreases as option moves ITM/OTM.
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        gamma = np.exp(-q * T) * stats.norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
        return gamma
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate vega (same for call and put).
        
        Vega measures sensitivity to volatility.
        Expressed as change per 1% volatility increase.
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        vega = S * np.exp(-q * T) * stats.norm.pdf(d1_val) * np.sqrt(T)
        return vega / 100  # Convention: per 1% vol change
    
    @staticmethod
    def call_theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate theta for call option.
        
        Theta measures time decay (usually negative).
        Expressed as change per day (divide by 365).
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholes.d2(S, K, T, r, sigma, q)
        
        term1 = -S * np.exp(-q * T) * stats.norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
        term2 = q * S * np.exp(-q * T) * stats.norm.cdf(d1_val)
        term3 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2_val)
        
        theta = term1 + term2 + term3
        return theta / 365  # Per day
    
    @staticmethod
    def put_theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate theta for put option."""
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholes.d2(S, K, T, r, sigma, q)
        
        term1 = -S * np.exp(-q * T) * stats.norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
        term2 = -q * S * np.exp(-q * T) * stats.norm.cdf(-d1_val)
        term3 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2_val)
        
        theta = term1 + term2 + term3
        return theta / 365  # Per day
    
    @staticmethod
    def call_rho(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate rho for call option.
        
        Rho measures sensitivity to interest rate.
        Expressed per 1% rate change.
        """
        if T <= 0:
            return 0.0
        
        d2_val = BlackScholes.d2(S, K, T, r, sigma, q)
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2_val)
        return rho / 100  # Per 1% rate change
    
    @staticmethod
    def put_rho(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate rho for put option."""
        if T <= 0:
            return 0.0
        
        d2_val = BlackScholes.d2(S, K, T, r, sigma, q)
        rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2_val)
        return rho / 100  # Per 1% rate change


class ImpliedVolatility:
    """
    Calculate implied volatility from market prices using numerical methods.
    """
    
    @staticmethod
    def call_iv(market_price: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> Optional[float]:
        """
        Calculate implied volatility for call option.
        
        Uses Brent's method for root finding.
        
        Args:
            market_price: Observed market price of call
            S, K, T, r, q: Black-Scholes parameters
        
        Returns:
            Implied volatility or None if not found
        """
        if T <= 0:
            return None
        
        # Objective function: BS price - market price
        def objective(sigma):
            try:
                return BlackScholes.call_price(S, K, T, r, sigma, q) - market_price
            except:
                return 1e10
        
        try:
            # Search between 1% and 500% volatility
            iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
            return iv
        except:
            return None
    
    @staticmethod
    def put_iv(market_price: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> Optional[float]:
        """Calculate implied volatility for put option."""
        if T <= 0:
            return None
        
        def objective(sigma):
            try:
                return BlackScholes.put_price(S, K, T, r, sigma, q) - market_price
            except:
                return 1e10
        
        try:
            iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
            return iv
        except:
            return None


class OptionAnalyzer:
    """
    Comprehensive option analysis combining pricing and Greeks.
    """
    
    @staticmethod
    def analyze_option(
        option_type: Literal["call", "put"],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict[str, float]:
        """
        Complete option analysis with price and all Greeks.
        
        Args:
            option_type: "call" or "put"
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
        
        Returns:
            Dictionary with price and Greeks
        """
        if option_type == "call":
            price = BlackScholes.call_price(S, K, T, r, sigma, q)
            delta = GreeksCalculator.call_delta(S, K, T, r, sigma, q)
            theta = GreeksCalculator.call_theta(S, K, T, r, sigma, q)
            rho = GreeksCalculator.call_rho(S, K, T, r, sigma, q)
        else:
            price = BlackScholes.put_price(S, K, T, r, sigma, q)
            delta = GreeksCalculator.put_delta(S, K, T, r, sigma, q)
            theta = GreeksCalculator.put_theta(S, K, T, r, sigma, q)
            rho = GreeksCalculator.put_rho(S, K, T, r, sigma, q)
        
        gamma = GreeksCalculator.gamma(S, K, T, r, sigma, q)
        vega = GreeksCalculator.vega(S, K, T, r, sigma, q)
        
        # Moneyness indicators
        moneyness = S / K
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        extrinsic = price - intrinsic
        
        return {
            "price": round(price, 4),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega": round(vega, 4),
            "theta": round(theta, 4),
            "rho": round(rho, 4),
            "intrinsic_value": round(intrinsic, 4),
            "time_value": round(extrinsic, 4),
            "moneyness": round(moneyness, 4),
            "moneyness_type": "ITM" if (option_type == "call" and S > K) or (option_type == "put" and S < K) else "OTM" if (option_type == "call" and S < K) or (option_type == "put" and S > K) else "ATM"
        }
