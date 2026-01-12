"""
Yield Curve Construction and Term Structure Analysis
Bootstrap, Nelson-Siegel-Svensson, cubic spline interpolation
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize, least_squares
import warnings
warnings.filterwarnings('ignore')


class YieldCurveBuilder:
    """
    Construct and analyze yield curves from market data.
    """
    
    def __init__(self):
        """Initialize yield curve builder."""
        self.curve = None
        self.method = None
    
    def bootstrap(self, 
                  instruments: pd.DataFrame,
                  settle_date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Bootstrap zero-coupon yield curve from instruments.
        
        Args:
            instruments: DataFrame with columns ['maturity', 'price', 'coupon', 'frequency']
            settle_date: Settlement date
        
        Returns:
            Zero-coupon yield curve
        """
        instruments = instruments.sort_values('maturity').reset_index(drop=True)
        zero_rates = {}
        
        for idx, row in instruments.iterrows():
            maturity = row['maturity']
            price = row['price']
            coupon = row.get('coupon', 0)
            frequency = row.get('frequency', 2)
            face_value = row.get('face_value', 100)
            
            # Zero coupon bond - direct calculation
            if coupon == 0:
                zero_rate = (face_value / price) ** (1 / maturity) - 1
                zero_rates[maturity] = zero_rate
                continue
            
            # Coupon bond - use previously bootstrapped rates
            periods = int(maturity * frequency)
            coupon_payment = (coupon * face_value) / frequency
            
            # Calculate PV of known cash flows
            pv_known = 0
            for t in range(1, periods):
                time = t / frequency
                if time in zero_rates:
                    rate = zero_rates[time]
                else:
                    # Interpolate
                    known_times = sorted(zero_rates.keys())
                    rate = np.interp(time, known_times, [zero_rates[k] for k in known_times])
                
                pv_known += coupon_payment / ((1 + rate) ** time)
            
            # Solve for zero rate at maturity
            remaining_pv = price - pv_known
            final_cash_flow = coupon_payment + face_value
            zero_rate = (final_cash_flow / remaining_pv) ** (1 / maturity) - 1
            zero_rates[maturity] = zero_rate
        
        self.curve = pd.Series(zero_rates).sort_index()
        self.method = 'bootstrap'
        return self.curve
    
    def nelson_siegel_svensson(self,
                               maturities: np.ndarray,
                               yields: np.ndarray,
                               initial_params: Optional[List[float]] = None) -> Dict:
        """
        Fit Nelson-Siegel-Svensson model to yield curve.
        
        NSS: y(t) = β0 + β1*(1-exp(-t/τ1))/(t/τ1) + 
                    β2*((1-exp(-t/τ1))/(t/τ1) - exp(-t/τ1)) + 
                    β3*((1-exp(-t/τ2))/(t/τ2) - exp(-t/τ2))
        
        Args:
            maturities: Array of maturities
            yields: Array of observed yields
            initial_params: Initial parameter guess [β0, β1, β2, β3, τ1, τ2]
        
        Returns:
            Dictionary with fitted parameters and curve
        """
        if initial_params is None:
            initial_params = [0.05, -0.02, 0.01, 0.01, 2.0, 5.0]
        
        def nss_model(t, beta0, beta1, beta2, beta3, tau1, tau2):
            """Nelson-Siegel-Svensson model."""
            term1 = beta0
            term2 = beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1))
            term3 = beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1))
            term4 = beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))
            return term1 + term2 + term3 + term4
        
        def objective(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            predicted = nss_model(maturities, beta0, beta1, beta2, beta3, tau1, tau2)
            return np.sum((predicted - yields) ** 2)
        
        # Bounds: reasonable ranges for parameters
        bounds = [
            (0, 0.2),      # beta0: level
            (-0.2, 0.2),   # beta1: slope
            (-0.2, 0.2),   # beta2: curvature 1
            (-0.2, 0.2),   # beta3: curvature 2
            (0.1, 10),     # tau1
            (0.1, 15)      # tau2
        ]
        
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        
        beta0, beta1, beta2, beta3, tau1, tau2 = result.x
        
        # Generate fitted curve
        t_fine = np.linspace(maturities.min(), maturities.max(), 100)
        fitted_yields = nss_model(t_fine, beta0, beta1, beta2, beta3, tau1, tau2)
        
        self.curve = pd.Series(fitted_yields, index=t_fine)
        self.method = 'nelson_siegel_svensson'
        
        return {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'beta3': beta3,
            'tau1': tau1,
            'tau2': tau2,
            'fitted_curve': self.curve,
            'rmse': np.sqrt(result.fun / len(yields)),
            'params': result.x
        }
    
    def cubic_spline(self,
                     maturities: np.ndarray,
                     yields: np.ndarray,
                     extrapolate: bool = True) -> pd.Series:
        """
        Fit cubic spline to yield curve.
        
        Args:
            maturities: Array of maturities
            yields: Array of observed yields
            extrapolate: Whether to extrapolate beyond data range
        
        Returns:
            Interpolated yield curve
        """
        # Sort by maturity
        sort_idx = np.argsort(maturities)
        maturities_sorted = maturities[sort_idx]
        yields_sorted = yields[sort_idx]
        
        # Create spline
        cs = CubicSpline(maturities_sorted, yields_sorted, extrapolate=extrapolate)
        
        # Generate fine grid
        t_fine = np.linspace(maturities_sorted.min(), maturities_sorted.max(), 100)
        fitted_yields = cs(t_fine)
        
        self.curve = pd.Series(fitted_yields, index=t_fine)
        self.method = 'cubic_spline'
        self.interpolator = cs
        
        return self.curve
    
    def get_rate(self, maturity: float) -> float:
        """
        Get rate for specific maturity from fitted curve.
        
        Args:
            maturity: Time to maturity in years
        
        Returns:
            Yield/rate
        """
        if self.curve is None:
            raise ValueError("Curve not fitted yet")
        
        if maturity in self.curve.index:
            return self.curve.loc[maturity]
        else:
            return np.interp(maturity, self.curve.index, self.curve.values)
    
    def forward_rates(self, step: float = 0.5) -> pd.Series:
        """
        Calculate instantaneous forward rates from zero curve.
        
        Args:
            step: Step size in years
        
        Returns:
            Forward rates
        """
        if self.curve is None:
            raise ValueError("Curve not fitted yet")
        
        maturities = self.curve.index
        zero_rates = self.curve.values
        
        forward_rates = []
        forward_maturities = []
        
        for i in range(len(maturities) - 1):
            t1, t2 = maturities[i], maturities[i + 1]
            r1, r2 = zero_rates[i], zero_rates[i + 1]
            
            # Forward rate
            f = ((1 + r2) ** t2 / (1 + r1) ** t1) ** (1 / (t2 - t1)) - 1
            forward_rates.append(f)
            forward_maturities.append((t1 + t2) / 2)
        
        return pd.Series(forward_rates, index=forward_maturities)
    
    def par_rates(self, maturities: np.ndarray, frequency: int = 2) -> pd.Series:
        """
        Calculate par rates (yields on par bonds) from zero curve.
        
        Args:
            maturities: Maturities for par rates
            frequency: Coupon frequency
        
        Returns:
            Par rates
        """
        if self.curve is None:
            raise ValueError("Curve not fitted yet")
        
        par_rates = []
        
        for maturity in maturities:
            periods = int(maturity * frequency)
            
            # Sum of discount factors
            df_sum = 0
            for t in range(1, periods + 1):
                time = t / frequency
                rate = self.get_rate(time)
                df = 1 / ((1 + rate) ** time)
                df_sum += df
            
            # Final discount factor
            final_rate = self.get_rate(maturity)
            df_final = 1 / ((1 + final_rate) ** maturity)
            
            # Par rate
            par_rate = (1 - df_final) / df_sum
            par_rates.append(par_rate * frequency)  # Annualize
        
        return pd.Series(par_rates, index=maturities)


class TermStructure:
    """
    Term structure analysis and decomposition.
    """
    
    @staticmethod
    def calculate_level_slope_curvature(yields: pd.Series) -> Dict:
        """
        Decompose yield curve into level, slope, and curvature factors.
        
        Args:
            yields: Yield curve (indexed by maturity)
        
        Returns:
            Dictionary with LSC factors
        """
        maturities = yields.index.values
        rates = yields.values
        
        # Level: average of all rates
        level = rates.mean()
        
        # Slope: difference between long and short end
        slope = rates[-1] - rates[0]
        
        # Curvature: 2 * medium - short - long
        mid_idx = len(rates) // 2
        curvature = 2 * rates[mid_idx] - rates[0] - rates[-1]
        
        # Calculate loadings via PCA if more sophisticated decomposition needed
        from sklearn.decomposition import PCA
        
        if len(rates) > 3:
            pca = PCA(n_components=3)
            rates_reshaped = rates.reshape(-1, 1)
            pca.fit(rates_reshaped)
            
            explained_var = pca.explained_variance_ratio_
        else:
            explained_var = [1.0, 0.0, 0.0]
        
        return {
            'level': level,
            'slope': slope,
            'curvature': curvature,
            'explained_variance_level': explained_var[0] if len(explained_var) > 0 else 1.0,
            'explained_variance_slope': explained_var[1] if len(explained_var) > 1 else 0.0,
            'explained_variance_curvature': explained_var[2] if len(explained_var) > 2 else 0.0
        }
    
    @staticmethod
    def detect_inversion(yields: pd.Series,
                        short_maturity: float = 2.0,
                        long_maturity: float = 10.0) -> bool:
        """
        Detect yield curve inversion.
        
        Args:
            yields: Yield curve
            short_maturity: Short-term maturity
            long_maturity: Long-term maturity
        
        Returns:
            True if inverted
        """
        # Interpolate if exact maturities not available
        if short_maturity not in yields.index:
            short_rate = np.interp(short_maturity, yields.index, yields.values)
        else:
            short_rate = yields.loc[short_maturity]
        
        if long_maturity not in yields.index:
            long_rate = np.interp(long_maturity, yields.index, yields.values)
        else:
            long_rate = yields.loc[long_maturity]
        
        return long_rate < short_rate
    
    @staticmethod
    def calculate_steepness(yields: pd.Series) -> float:
        """
        Calculate yield curve steepness (10Y - 2Y).
        
        Args:
            yields: Yield curve
        
        Returns:
            Steepness in basis points
        """
        rate_10y = np.interp(10, yields.index, yields.values)
        rate_2y = np.interp(2, yields.index, yields.values)
        
        return (rate_10y - rate_2y) * 10000  # In basis points
    
    @staticmethod
    def butterfly_spread(yields: pd.Series,
                        short: float = 2.0,
                        medium: float = 5.0,
                        long: float = 10.0) -> float:
        """
        Calculate butterfly spread.
        
        Args:
            yields: Yield curve
            short: Short maturity
            medium: Medium maturity
            long: Long maturity
        
        Returns:
            Butterfly spread
        """
        rate_short = np.interp(short, yields.index, yields.values)
        rate_medium = np.interp(medium, yields.index, yields.values)
        rate_long = np.interp(long, yields.index, yields.values)
        
        # Butterfly = short + long - 2 * medium
        butterfly = rate_short + rate_long - 2 * rate_medium
        
        return butterfly * 10000  # In basis points
    
    @staticmethod
    def term_premium(yields: pd.Series,
                    expected_short_rates: Optional[pd.Series] = None) -> pd.Series:
        """
        Estimate term premium.
        
        Args:
            yields: Observed yield curve
            expected_short_rates: Expected average short rates (if None, uses simple model)
        
        Returns:
            Term premium series
        """
        if expected_short_rates is None:
            # Simple model: assume short rate expectations follow current short rate
            short_rate = yields.iloc[0] if len(yields) > 0 else 0
            expected_short_rates = pd.Series([short_rate] * len(yields), index=yields.index)
        
        term_premium = yields - expected_short_rates
        return term_premium
