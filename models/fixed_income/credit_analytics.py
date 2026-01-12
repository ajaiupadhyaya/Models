"""
Credit Risk Analytics
Credit spreads, default probability, credit ratings, CDS pricing
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')


class CreditSpreadAnalyzer:
    """
    Credit spread analysis and decomposition.
    """
    
    @staticmethod
    def calculate_credit_spread(corporate_yield: float,
                               treasury_yield: float) -> float:
        """
        Calculate credit spread.
        
        Args:
            corporate_yield: Corporate bond yield
            treasury_yield: Risk-free treasury yield
        
        Returns:
            Credit spread in basis points
        """
        return (corporate_yield - treasury_yield) * 10000
    
    @staticmethod
    def option_adjusted_spread(bond_price: float,
                              treasury_curve: pd.Series,
                              cash_flows: pd.DataFrame,
                              volatility: float = 0.10) -> float:
        """
        Calculate option-adjusted spread (OAS).
        
        Args:
            bond_price: Market price of bond
            treasury_curve: Risk-free yield curve
            cash_flows: DataFrame with 'time' and 'amount' columns
            volatility: Interest rate volatility
        
        Returns:
            OAS in basis points
        """
        def price_with_spread(spread):
            pv = 0
            for _, cf in cash_flows.iterrows():
                time = cf['time']
                amount = cf['amount']
                
                # Get treasury rate for this maturity
                if time in treasury_curve.index:
                    treasury_rate = treasury_curve.loc[time]
                else:
                    treasury_rate = np.interp(time, treasury_curve.index, treasury_curve.values)
                
                discount_rate = treasury_rate + spread
                pv += amount / ((1 + discount_rate) ** time)
            
            return pv
        
        def objective(spread):
            return price_with_spread(spread) - bond_price
        
        try:
            oas = brentq(objective, -0.1, 0.5)
            return oas * 10000  # In basis points
        except:
            return np.nan
    
    @staticmethod
    def decompose_credit_spread(total_spread: float,
                               expected_loss: float,
                               recovery_rate: float = 0.40) -> Dict:
        """
        Decompose credit spread into expected loss and risk premium.
        
        Args:
            total_spread: Total credit spread
            expected_loss: Expected loss rate
            recovery_rate: Recovery rate
        
        Returns:
            Dictionary with spread components
        """
        # Expected loss component
        expected_loss_spread = expected_loss / (1 - recovery_rate)
        
        # Risk premium (residual)
        risk_premium = total_spread - expected_loss_spread
        
        return {
            'total_spread_bps': total_spread,
            'expected_loss_spread_bps': expected_loss_spread,
            'risk_premium_bps': risk_premium,
            'expected_loss_pct': expected_loss_spread / total_spread if total_spread > 0 else 0,
            'risk_premium_pct': risk_premium / total_spread if total_spread > 0 else 0
        }
    
    @staticmethod
    def z_spread_to_oas_adjustment(embedded_option_value: float,
                                   bond_price: float) -> float:
        """
        Calculate adjustment from Z-spread to OAS.
        
        Args:
            embedded_option_value: Value of embedded option
            bond_price: Bond price
        
        Returns:
            Adjustment in basis points
        """
        # OAS = Z-spread - option cost
        option_cost_pct = embedded_option_value / bond_price
        return -option_cost_pct * 10000


class DefaultProbability:
    """
    Default probability estimation and credit modeling.
    """
    
    @staticmethod
    def from_credit_spread(credit_spread: float,
                          recovery_rate: float = 0.40,
                          time_horizon: float = 1.0) -> float:
        """
        Estimate default probability from credit spread.
        
        Using simplified Merton model:
        credit_spread ≈ PD * (1 - RR)
        
        Args:
            credit_spread: Credit spread (decimal, not bps)
            recovery_rate: Recovery rate
            time_horizon: Time horizon in years
        
        Returns:
            Probability of default
        """
        pd = credit_spread / (1 - recovery_rate) / time_horizon
        return min(pd, 1.0)  # Cap at 100%
    
    @staticmethod
    def merton_model(asset_value: float,
                    debt_face_value: float,
                    volatility: float,
                    risk_free_rate: float,
                    time_horizon: float) -> Dict:
        """
        Merton structural credit model.
        
        Default occurs when asset value < debt at maturity.
        
        Args:
            asset_value: Current asset value
            debt_face_value: Face value of debt
            volatility: Asset volatility
            risk_free_rate: Risk-free rate
            time_horizon: Time to maturity
        
        Returns:
            Dictionary with default probability and distance to default
        """
        # Distance to default
        d2 = (np.log(asset_value / debt_face_value) + 
              (risk_free_rate - 0.5 * volatility**2) * time_horizon) / (
              volatility * np.sqrt(time_horizon))
        
        # Probability of default
        pd = norm.cdf(-d2)
        
        # Distance to default (in standard deviations)
        dd = d2
        
        # Equity value (call option on assets)
        d1 = d2 + volatility * np.sqrt(time_horizon)
        equity_value = asset_value * norm.cdf(d1) - debt_face_value * np.exp(
            -risk_free_rate * time_horizon) * norm.cdf(d2)
        
        # Credit spread approximation
        credit_spread = -np.log(1 - pd) / time_horizon
        
        return {
            'default_probability': pd,
            'distance_to_default': dd,
            'equity_value': equity_value,
            'implied_credit_spread': credit_spread,
            'd1': d1,
            'd2': d2
        }
    
    @staticmethod
    def hazard_rate_model(hazard_rate: float,
                         time_horizon: float,
                         recovery_rate: float = 0.40) -> Dict:
        """
        Reduced-form hazard rate model.
        
        Args:
            hazard_rate: Constant hazard rate (intensity)
            time_horizon: Time horizon
            recovery_rate: Recovery rate
        
        Returns:
            Dictionary with survival probability and credit metrics
        """
        # Survival probability
        survival_prob = np.exp(-hazard_rate * time_horizon)
        
        # Default probability
        default_prob = 1 - survival_prob
        
        # Expected loss
        expected_loss = default_prob * (1 - recovery_rate)
        
        # Credit spread
        credit_spread = hazard_rate * (1 - recovery_rate)
        
        return {
            'hazard_rate': hazard_rate,
            'survival_probability': survival_prob,
            'default_probability': default_prob,
            'expected_loss': expected_loss,
            'credit_spread': credit_spread,
            'credit_spread_bps': credit_spread * 10000
        }
    
    @staticmethod
    def cumulative_default_probability(annual_pd: float,
                                      years: int) -> List[float]:
        """
        Calculate cumulative default probability over time.
        
        Args:
            annual_pd: Annual probability of default
            years: Number of years
        
        Returns:
            List of cumulative default probabilities
        """
        cumulative_pds = []
        survival_prob = 1.0
        
        for year in range(1, years + 1):
            survival_prob *= (1 - annual_pd)
            cumulative_pd = 1 - survival_prob
            cumulative_pds.append(cumulative_pd)
        
        return cumulative_pds
    
    @staticmethod
    def transition_matrix_default_prob(transition_matrix: np.ndarray,
                                      initial_rating: int,
                                      time_horizon: int) -> float:
        """
        Calculate default probability using rating transition matrix.
        
        Args:
            transition_matrix: Rating transition matrix (rows/cols are ratings, last is default)
            initial_rating: Initial rating index
            time_horizon: Time horizon in years
        
        Returns:
            Probability of default
        """
        # Raise matrix to power for multi-year horizon
        matrix_t = np.linalg.matrix_power(transition_matrix, time_horizon)
        
        # Default probability is transition to default state (last column)
        pd = matrix_t[initial_rating, -1]
        
        return pd


class CDSPricer:
    """
    Credit Default Swap pricing and analytics.
    """
    
    def __init__(self,
                 notional: float = 1000000,
                 recovery_rate: float = 0.40,
                 spread: Optional[float] = None):
        """
        Initialize CDS pricer.
        
        Args:
            notional: Notional amount
            recovery_rate: Recovery rate
            spread: CDS spread in basis points
        """
        self.notional = notional
        self.recovery_rate = recovery_rate
        self.spread = spread / 10000 if spread is not None else None
    
    def calculate_spread(self,
                        hazard_rate: float,
                        risk_free_curve: pd.Series,
                        maturity: float,
                        frequency: int = 4) -> float:
        """
        Calculate fair CDS spread.
        
        Args:
            hazard_rate: Credit hazard rate
            risk_free_curve: Risk-free discount curve
            maturity: CDS maturity in years
            frequency: Payment frequency (4 = quarterly)
        
        Returns:
            Fair CDS spread in basis points
        """
        periods = int(maturity * frequency)
        dt = 1 / frequency
        
        # Calculate expected payment leg (premium leg)
        premium_leg = 0
        # Calculate expected protection leg
        protection_leg = 0
        
        survival_prob = 1.0
        
        for t in range(1, periods + 1):
            time = t * dt
            
            # Get discount factor
            if time in risk_free_curve.index:
                rate = risk_free_curve.loc[time]
            else:
                rate = np.interp(time, risk_free_curve.index, risk_free_curve.values)
            
            discount_factor = np.exp(-rate * time)
            
            # Survival probability
            survival_prob_t = np.exp(-hazard_rate * time)
            
            # Default probability in this period
            default_prob_t = survival_prob - survival_prob_t
            
            # Premium leg: spread * notional * survival_prob * discount_factor
            premium_leg += survival_prob_t * discount_factor * dt
            
            # Protection leg: (1 - recovery) * notional * default_prob * discount_factor
            protection_leg += (1 - self.recovery_rate) * default_prob_t * discount_factor
            
            survival_prob = survival_prob_t
        
        # Fair spread
        if premium_leg > 0:
            spread = protection_leg / premium_leg
        else:
            spread = 0
        
        return spread * 10000  # In basis points
    
    def upfront_payment(self,
                       market_spread: float,
                       coupon: float,
                       hazard_rate: float,
                       risk_free_curve: pd.Series,
                       maturity: float) -> float:
        """
        Calculate upfront payment for CDS.
        
        Args:
            market_spread: Market CDS spread (bps)
            coupon: Fixed coupon rate (bps)
            hazard_rate: Hazard rate
            risk_free_curve: Risk-free curve
            maturity: Maturity in years
        
        Returns:
            Upfront payment as percentage of notional
        """
        market_spread = market_spread / 10000
        coupon = coupon / 10000
        
        # Calculate PV of spread differential
        spread_diff = market_spread - coupon
        
        periods = int(maturity * 4)  # Quarterly
        dt = 0.25
        
        pv = 0
        for t in range(1, periods + 1):
            time = t * dt
            
            # Discount and survival
            if time in risk_free_curve.index:
                rate = risk_free_curve.loc[time]
            else:
                rate = np.interp(time, risk_free_curve.index, risk_free_curve.values)
            
            discount_factor = np.exp(-rate * time)
            survival_prob = np.exp(-hazard_rate * time)
            
            pv += spread_diff * survival_prob * discount_factor * dt
        
        return pv * self.notional
    
    def risky_pv01(self,
                  hazard_rate: float,
                  risk_free_curve: pd.Series,
                  maturity: float,
                  frequency: int = 4) -> float:
        """
        Calculate risky PV01 (present value of 1bp annuity).
        
        Args:
            hazard_rate: Hazard rate
            risk_free_curve: Risk-free curve
            maturity: Maturity in years
            frequency: Payment frequency
        
        Returns:
            Risky PV01
        """
        periods = int(maturity * frequency)
        dt = 1 / frequency
        
        pv01 = 0
        
        for t in range(1, periods + 1):
            time = t * dt
            
            if time in risk_free_curve.index:
                rate = risk_free_curve.loc[time]
            else:
                rate = np.interp(time, risk_free_curve.index, risk_free_curve.values)
            
            discount_factor = np.exp(-rate * time)
            survival_prob = np.exp(-hazard_rate * time)
            
            pv01 += survival_prob * discount_factor * dt
        
        return pv01 * 0.0001  # 1 basis point
    
    def cs01(self,
            hazard_rate: float,
            risk_free_curve: pd.Series,
            maturity: float) -> float:
        """
        Calculate CS01 (spread sensitivity).
        
        Args:
            hazard_rate: Hazard rate
            risk_free_curve: Risk-free curve
            maturity: Maturity
        
        Returns:
            CS01 (P&L change for 1bp spread move)
        """
        rpv01 = self.risky_pv01(hazard_rate, risk_free_curve, maturity)
        return rpv01 * self.notional


class CreditRatingModel:
    """
    Credit rating analytics and mapping.
    """
    
    # Standard rating mapping
    RATING_MAP = {
        'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
        'A+': 5, 'A': 6, 'A-': 7,
        'BBB+': 8, 'BBB': 9, 'BBB-': 10,
        'BB+': 11, 'BB': 12, 'BB-': 13,
        'B+': 14, 'B': 15, 'B-': 16,
        'CCC+': 17, 'CCC': 18, 'CCC-': 19,
        'CC': 20, 'C': 21, 'D': 22
    }
    
    # Typical default probabilities by rating (annual)
    TYPICAL_PDS = {
        'AAA': 0.0002, 'AA+': 0.0003, 'AA': 0.0004, 'AA-': 0.0006,
        'A+': 0.0010, 'A': 0.0015, 'A-': 0.0025,
        'BBB+': 0.0040, 'BBB': 0.0060, 'BBB-': 0.0100,
        'BB+': 0.0200, 'BB': 0.0350, 'BB-': 0.0600,
        'B+': 0.1000, 'B': 0.1500, 'B-': 0.2500,
        'CCC+': 0.3500, 'CCC': 0.4500, 'CCC-': 0.6000,
        'CC': 0.7500, 'C': 0.9000, 'D': 1.0000
    }
    
    @staticmethod
    def rating_to_numeric(rating: str) -> int:
        """Convert rating to numeric score."""
        return CreditRatingModel.RATING_MAP.get(rating, 0)
    
    @staticmethod
    def numeric_to_rating(score: int) -> str:
        """Convert numeric score to rating."""
        reverse_map = {v: k for k, v in CreditRatingModel.RATING_MAP.items()}
        return reverse_map.get(score, 'NR')
    
    @staticmethod
    def get_default_probability(rating: str) -> float:
        """Get typical default probability for rating."""
        return CreditRatingModel.TYPICAL_PDS.get(rating, 0.5)
    
    @staticmethod
    def is_investment_grade(rating: str) -> bool:
        """Check if rating is investment grade (BBB- or better)."""
        score = CreditRatingModel.rating_to_numeric(rating)
        return 1 <= score <= 10
    
    @staticmethod
    def rating_difference(rating1: str, rating2: str) -> int:
        """Calculate notches between two ratings."""
        score1 = CreditRatingModel.rating_to_numeric(rating1)
        score2 = CreditRatingModel.rating_to_numeric(rating2)
        return abs(score1 - score2)
