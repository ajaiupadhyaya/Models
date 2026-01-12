"""
Comprehensive Bond Analytics
Duration, convexity, yield calculations, bond pricing
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')


class BondPricer:
    """
    Institutional-grade bond pricing and analytics.
    """
    
    def __init__(self,
                 face_value: float = 1000,
                 coupon_rate: float = 0.05,
                 years_to_maturity: float = 10,
                 frequency: int = 2,  # Semi-annual = 2
                 day_count: str = '30/360'):
        """
        Initialize bond pricer.
        
        Args:
            face_value: Face/par value of bond
            coupon_rate: Annual coupon rate
            years_to_maturity: Years to maturity
            frequency: Coupon payment frequency per year
            day_count: Day count convention (30/360, Actual/360, Actual/365, Actual/Actual)
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.frequency = frequency
        self.day_count = day_count
        self.coupon_payment = (face_value * coupon_rate) / frequency
        self.periods = int(years_to_maturity * frequency)
    
    def price(self, ytm: float) -> float:
        """
        Calculate bond price given yield to maturity.
        
        Args:
            ytm: Yield to maturity (annual)
        
        Returns:
            Bond price
        """
        if self.periods == 0:
            return self.face_value
        
        # Discount rate per period
        r = ytm / self.frequency
        
        # Present value of coupons
        pv_coupons = sum([
            self.coupon_payment / ((1 + r) ** t)
            for t in range(1, self.periods + 1)
        ])
        
        # Present value of face value
        pv_face = self.face_value / ((1 + r) ** self.periods)
        
        return pv_coupons + pv_face
    
    def yield_to_maturity(self, price: float, initial_guess: float = 0.05) -> float:
        """
        Calculate yield to maturity given bond price.
        
        Args:
            price: Current bond price
            initial_guess: Initial guess for YTM
        
        Returns:
            Yield to maturity
        """
        def objective(ytm):
            return self.price(ytm) - price
        
        try:
            ytm = brentq(objective, -0.5, 1.0)
            return ytm
        except:
            # Fallback to Newton's method
            from scipy.optimize import newton
            try:
                return newton(objective, initial_guess)
            except:
                return np.nan
    
    def current_yield(self, price: float) -> float:
        """
        Calculate current yield.
        
        Args:
            price: Current bond price
        
        Returns:
            Current yield
        """
        annual_coupon = self.face_value * self.coupon_rate
        return annual_coupon / price
    
    def macaulay_duration(self, ytm: float) -> float:
        """
        Calculate Macaulay duration.
        
        Args:
            ytm: Yield to maturity
        
        Returns:
            Macaulay duration in years
        """
        r = ytm / self.frequency
        price = self.price(ytm)
        
        # Weighted average time to receive cash flows
        weighted_cash_flows = 0
        
        for t in range(1, self.periods + 1):
            time_years = t / self.frequency
            cash_flow = self.coupon_payment
            if t == self.periods:
                cash_flow += self.face_value
            
            pv = cash_flow / ((1 + r) ** t)
            weighted_cash_flows += time_years * pv
        
        return weighted_cash_flows / price
    
    def modified_duration(self, ytm: float) -> float:
        """
        Calculate modified duration.
        
        Args:
            ytm: Yield to maturity
        
        Returns:
            Modified duration
        """
        mac_dur = self.macaulay_duration(ytm)
        return mac_dur / (1 + ytm / self.frequency)
    
    def dollar_duration(self, ytm: float) -> float:
        """
        Calculate dollar duration (DV01).
        
        Args:
            ytm: Yield to maturity
        
        Returns:
            Dollar duration (price change for 1bp move)
        """
        price = self.price(ytm)
        mod_dur = self.modified_duration(ytm)
        return -mod_dur * price * 0.0001  # 1 basis point
    
    def convexity(self, ytm: float) -> float:
        """
        Calculate bond convexity.
        
        Args:
            ytm: Yield to maturity
        
        Returns:
            Convexity
        """
        r = ytm / self.frequency
        price = self.price(ytm)
        
        weighted_cash_flows = 0
        
        for t in range(1, self.periods + 1):
            cash_flow = self.coupon_payment
            if t == self.periods:
                cash_flow += self.face_value
            
            pv = cash_flow / ((1 + r) ** t)
            weighted_cash_flows += t * (t + 1) * pv
        
        convexity = weighted_cash_flows / (price * (1 + r) ** 2 * self.frequency ** 2)
        return convexity
    
    def price_change_estimate(self, ytm: float, yield_change: float) -> Dict[str, float]:
        """
        Estimate price change for yield change using duration and convexity.
        
        Args:
            ytm: Current yield to maturity
            yield_change: Change in yield (e.g., 0.01 for 1%)
        
        Returns:
            Dictionary with estimates
        """
        current_price = self.price(ytm)
        mod_dur = self.modified_duration(ytm)
        conv = self.convexity(ytm)
        
        # Duration approximation
        delta_P_duration = -mod_dur * current_price * yield_change
        
        # Duration + Convexity approximation
        delta_P_full = (
            -mod_dur * current_price * yield_change +
            0.5 * conv * current_price * (yield_change ** 2)
        )
        
        # Actual price change
        new_price = self.price(ytm + yield_change)
        actual_change = new_price - current_price
        
        return {
            'current_price': current_price,
            'new_ytm': ytm + yield_change,
            'new_price_actual': new_price,
            'actual_change': actual_change,
            'duration_estimate': delta_P_duration,
            'duration_convexity_estimate': delta_P_full,
            'error_duration_only': abs(actual_change - delta_P_duration),
            'error_with_convexity': abs(actual_change - delta_P_full)
        }
    
    def accrued_interest(self, settlement_date: datetime, last_coupon_date: datetime) -> float:
        """
        Calculate accrued interest.
        
        Args:
            settlement_date: Settlement date
            last_coupon_date: Last coupon payment date
        
        Returns:
            Accrued interest
        """
        days_since_coupon = (settlement_date - last_coupon_date).days
        days_in_period = 365.25 / self.frequency
        
        if self.day_count == '30/360':
            days_in_period = 180 if self.frequency == 2 else 360 / self.frequency
        
        accrued = (days_since_coupon / days_in_period) * self.coupon_payment
        return accrued
    
    def clean_price(self, dirty_price: float, accrued: float) -> float:
        """
        Calculate clean price from dirty price.
        
        Args:
            dirty_price: Dirty (full) price
            accrued: Accrued interest
        
        Returns:
            Clean price
        """
        return dirty_price - accrued
    
    def z_spread(self, price: float, spot_curve: pd.Series) -> float:
        """
        Calculate Z-spread over spot curve.
        
        Args:
            price: Bond price
            spot_curve: Spot rate curve (indexed by maturity)
        
        Returns:
            Z-spread in basis points
        """
        def price_with_spread(spread):
            pv = 0
            for t in range(1, self.periods + 1):
                time_years = t / self.frequency
                cash_flow = self.coupon_payment
                if t == self.periods:
                    cash_flow += self.face_value
                
                # Interpolate spot rate
                if time_years in spot_curve.index:
                    spot_rate = spot_curve.loc[time_years]
                else:
                    spot_rate = np.interp(time_years, spot_curve.index, spot_curve.values)
                
                discount_rate = spot_rate + spread
                pv += cash_flow / ((1 + discount_rate / self.frequency) ** t)
            
            return pv
        
        def objective(spread):
            return price_with_spread(spread) - price
        
        try:
            z_spread = brentq(objective, -0.1, 0.5)
            return z_spread * 10000  # Convert to basis points
        except:
            return np.nan
    
    def get_cash_flows(self) -> pd.DataFrame:
        """
        Get all cash flows.
        
        Returns:
            DataFrame with cash flow schedule
        """
        cash_flows = []
        
        for t in range(1, self.periods + 1):
            time_years = t / self.frequency
            cash_flow = self.coupon_payment
            
            if t == self.periods:
                cash_flow += self.face_value
            
            cash_flows.append({
                'period': t,
                'time_years': time_years,
                'cash_flow': cash_flow,
                'type': 'Coupon' if t < self.periods else 'Coupon + Principal'
            })
        
        return pd.DataFrame(cash_flows)
    
    def summary(self, ytm: float, price: Optional[float] = None) -> Dict:
        """
        Get comprehensive bond analytics summary.
        
        Args:
            ytm: Yield to maturity
            price: Optional price (if not provided, calculated from YTM)
        
        Returns:
            Dictionary with all analytics
        """
        if price is None:
            price = self.price(ytm)
        else:
            ytm = self.yield_to_maturity(price)
        
        return {
            'face_value': self.face_value,
            'coupon_rate': self.coupon_rate,
            'years_to_maturity': self.years_to_maturity,
            'frequency': self.frequency,
            'price': price,
            'yield_to_maturity': ytm,
            'current_yield': self.current_yield(price),
            'macaulay_duration': self.macaulay_duration(ytm),
            'modified_duration': self.modified_duration(ytm),
            'dollar_duration': self.dollar_duration(ytm),
            'convexity': self.convexity(ytm),
            'annual_coupon': self.face_value * self.coupon_rate
        }


class BondPortfolio:
    """
    Bond portfolio analytics and risk management.
    """
    
    def __init__(self, bonds: List[BondPricer], weights: List[float]):
        """
        Initialize bond portfolio.
        
        Args:
            bonds: List of BondPricer objects
            weights: Portfolio weights
        """
        self.bonds = bonds
        self.weights = np.array(weights)
        
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def portfolio_duration(self, ytms: List[float]) -> float:
        """
        Calculate portfolio duration.
        
        Args:
            ytms: List of YTMs for each bond
        
        Returns:
            Portfolio duration
        """
        durations = [bond.modified_duration(ytm) for bond, ytm in zip(self.bonds, ytms)]
        return np.dot(self.weights, durations)
    
    def portfolio_convexity(self, ytms: List[float]) -> float:
        """
        Calculate portfolio convexity.
        
        Args:
            ytms: List of YTMs for each bond
        
        Returns:
            Portfolio convexity
        """
        convexities = [bond.convexity(ytm) for bond, ytm in zip(self.bonds, ytms)]
        return np.dot(self.weights, convexities)
    
    def portfolio_value(self, ytms: List[float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            ytms: List of YTMs for each bond
        
        Returns:
            Portfolio value
        """
        values = [bond.price(ytm) * weight 
                 for bond, ytm, weight in zip(self.bonds, ytms, self.weights)]
        return sum(values)
    
    def interest_rate_risk(self, ytms: List[float], rate_shock: float = 0.01) -> Dict:
        """
        Assess interest rate risk.
        
        Args:
            ytms: List of YTMs for each bond
            rate_shock: Interest rate shock (e.g., 0.01 for 100bp)
        
        Returns:
            Dictionary with risk metrics
        """
        current_value = self.portfolio_value(ytms)
        shocked_ytms = [ytm + rate_shock for ytm in ytms]
        shocked_value = self.portfolio_value(shocked_ytms)
        
        value_change = shocked_value - current_value
        pct_change = value_change / current_value
        
        return {
            'current_value': current_value,
            'shocked_value': shocked_value,
            'value_change': value_change,
            'percent_change': pct_change,
            'rate_shock_bps': rate_shock * 10000,
            'portfolio_duration': self.portfolio_duration(ytms),
            'portfolio_convexity': self.portfolio_convexity(ytms)
        }
    
    def key_rate_durations(self, ytms: List[float], key_maturities: List[float]) -> Dict:
        """
        Calculate key rate durations.
        
        Args:
            ytms: List of YTMs for each bond
            key_maturities: Key maturity points
        
        Returns:
            Dictionary of key rate durations
        """
        key_rate_durs = {}
        shock = 0.0001  # 1 basis point
        
        for maturity in key_maturities:
            # Find bonds closest to this maturity
            base_value = self.portfolio_value(ytms)
            
            # Shock rates near this maturity
            shocked_ytms = ytms.copy()
            for i, bond in enumerate(self.bonds):
                if abs(bond.years_to_maturity - maturity) < 1.0:
                    shocked_ytms[i] += shock
            
            shocked_value = self.portfolio_value(shocked_ytms)
            krd = -(shocked_value - base_value) / (base_value * shock)
            
            key_rate_durs[f'{maturity}Y'] = krd
        
        return key_rate_durs
