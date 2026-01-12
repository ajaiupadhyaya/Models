"""
Black-Scholes-Merton Options Pricing Model
Complete implementation with Greeks calculation.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class BlackScholes:
    """
    Black-Scholes-Merton options pricing model.
    """
    
    @staticmethod
    def calculate_d1_d2(S: float, 
                       K: float, 
                       T: float, 
                       r: float, 
                       sigma: float) -> tuple:
        """
        Calculate d1 and d2 parameters.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Tuple of (d1, d2)
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def call_price(S: float, 
                   K: float, 
                   T: float, 
                   r: float, 
                   sigma: float,
                   dividend_yield: float = 0.0) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Dividend yield
        
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r - dividend_yield, sigma)
        
        call_price = (S * np.exp(-dividend_yield * T) * norm.cdf(d1) 
                     - K * np.exp(-r * T) * norm.cdf(d2))
        
        return call_price
    
    @staticmethod
    def put_price(S: float, 
                  K: float, 
                  T: float, 
                  r: float, 
                  sigma: float,
                  dividend_yield: float = 0.0) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Dividend yield
        
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        # Use put-call parity
        call = BlackScholes.call_price(S, K, T, r, sigma, dividend_yield)
        put = call - S * np.exp(-dividend_yield * T) + K * np.exp(-r * T)
        
        return put
    
    @staticmethod
    def delta(S: float, 
             K: float, 
             T: float, 
             r: float, 
             sigma: float,
             option_type: str = 'call',
             dividend_yield: float = 0.0) -> float:
        """
        Calculate option delta (price sensitivity to underlying).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
        
        Returns:
            Delta
        """
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r - dividend_yield, sigma)
        
        if option_type == 'call':
            return np.exp(-dividend_yield * T) * norm.cdf(d1)
        else:  # put
            return -np.exp(-dividend_yield * T) * norm.cdf(-d1)
    
    @staticmethod
    def gamma(S: float, 
             K: float, 
             T: float, 
             r: float, 
             sigma: float,
             dividend_yield: float = 0.0) -> float:
        """
        Calculate option gamma (delta sensitivity).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Dividend yield
        
        Returns:
            Gamma
        """
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r - dividend_yield, sigma)
        gamma = (np.exp(-dividend_yield * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        return gamma
    
    @staticmethod
    def vega(S: float, 
            K: float, 
            T: float, 
            r: float, 
            sigma: float,
            dividend_yield: float = 0.0) -> float:
        """
        Calculate option vega (volatility sensitivity).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Dividend yield
        
        Returns:
            Vega (per 1% change in volatility)
        """
        d1, _ = BlackScholes.calculate_d1_d2(S, K, T, r - dividend_yield, sigma)
        vega = S * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
        return vega
    
    @staticmethod
    def theta(S: float, 
             K: float, 
             T: float, 
             r: float, 
             sigma: float,
             option_type: str = 'call',
             dividend_yield: float = 0.0) -> float:
        """
        Calculate option theta (time decay).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
        
        Returns:
            Theta (per day)
        """
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r - dividend_yield, sigma)
        
        if option_type == 'call':
            theta = (-(S * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(d1)) / 365
        else:  # put
            theta = (-(S * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)
                    - dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(-d1)) / 365
        
        return theta
    
    @staticmethod
    def rho(S: float, 
           K: float, 
           T: float, 
           r: float, 
           sigma: float,
           option_type: str = 'call',
           dividend_yield: float = 0.0) -> float:
        """
        Calculate option rho (interest rate sensitivity).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
        
        Returns:
            Rho (per 1% change in rate)
        """
        d1, d2 = BlackScholes.calculate_d1_d2(S, K, T, r - dividend_yield, sigma)
        
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return rho
    
    @staticmethod
    def implied_volatility(market_price: float,
                          S: float,
                          K: float,
                          T: float,
                          r: float,
                          option_type: str = 'call',
                          dividend_yield: float = 0.0,
                          initial_guess: float = 0.2) -> float:
        """
        Calculate implied volatility from market price.
        
        Args:
            market_price: Market price of option
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            initial_guess: Initial volatility guess
        
        Returns:
            Implied volatility
        """
        from scipy.optimize import brentq
        
        def price_diff(sigma):
            if option_type == 'call':
                price = BlackScholes.call_price(S, K, T, r, sigma, dividend_yield)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma, dividend_yield)
            return price - market_price
        
        try:
            iv = brentq(price_diff, 0.001, 5.0)
            return iv
        except:
            return np.nan
    
    @staticmethod
    def get_all_greeks(S: float,
                      K: float,
                      T: float,
                      r: float,
                      sigma: float,
                      option_type: str = 'call',
                      dividend_yield: float = 0.0) -> dict:
        """
        Calculate all option Greeks.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
        
        Returns:
            Dictionary with all Greeks
        """
        return {
            'delta': BlackScholes.delta(S, K, T, r, sigma, option_type, dividend_yield),
            'gamma': BlackScholes.gamma(S, K, T, r, sigma, dividend_yield),
            'vega': BlackScholes.vega(S, K, T, r, sigma, dividend_yield),
            'theta': BlackScholes.theta(S, K, T, r, sigma, option_type, dividend_yield),
            'rho': BlackScholes.rho(S, K, T, r, sigma, option_type, dividend_yield)
        }
