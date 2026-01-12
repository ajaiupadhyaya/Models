"""
Advanced Macroeconomic Models
DSGE components, yield curve models, term structure analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class YieldCurveModel:
    """
    Advanced yield curve modeling and analysis.
    """
    
    def __init__(self):
        """Initialize yield curve model."""
        pass
    
    def nelson_siegel(self, 
                     t: np.ndarray,
                     beta0: float,
                     beta1: float,
                     beta2: float,
                     tau: float) -> np.ndarray:
        """
        Nelson-Siegel yield curve model.
        
        Args:
            t: Time to maturity
            beta0: Long-term rate
            beta1: Short-term component
            beta2: Medium-term component
            tau: Decay parameter
        
        Returns:
            Yield curve
        """
        term1 = beta0
        term2 = beta1 * ((1 - np.exp(-t / tau)) / (t / tau))
        term3 = beta2 * (((1 - np.exp(-t / tau)) / (t / tau)) - np.exp(-t / tau))
        
        return term1 + term2 + term3
    
    def fit_nelson_siegel(self, 
                         maturities: np.ndarray,
                         yields: np.ndarray,
                         initial_params: Optional[List[float]] = None) -> Dict:
        """
        Fit Nelson-Siegel model to observed yields.
        
        Args:
            maturities: Maturities in years
            yields: Observed yields
            initial_params: Initial parameter guess
        
        Returns:
            Dictionary with fitted parameters
        """
        if initial_params is None:
            initial_params = [0.05, -0.02, 0.01, 2.0]
        
        def objective(params):
            beta0, beta1, beta2, tau = params
            predicted = self.nelson_siegel(maturities, beta0, beta1, beta2, tau)
            return np.sum((predicted - yields) ** 2)
        
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         bounds=[(0, 0.2), (-0.1, 0.1), (-0.1, 0.1), (0.1, 10)])
        
        beta0, beta1, beta2, tau = result.x
        
        return {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'tau': tau,
            'fitted_yields': self.nelson_siegel(maturities, beta0, beta1, beta2, tau),
            'rmse': np.sqrt(result.fun / len(yields))
        }
    
    def calculate_forward_rates(self,
                               spot_rates: pd.Series,
                               maturities: np.ndarray) -> np.ndarray:
        """
        Calculate forward rates from spot rates.
        
        Args:
            spot_rates: Spot rates
            maturities: Maturities
        
        Returns:
            Forward rates
        """
        forward_rates = np.zeros(len(maturities) - 1)
        
        for i in range(len(maturities) - 1):
            t1, t2 = maturities[i], maturities[i+1]
            r1, r2 = spot_rates.iloc[i], spot_rates.iloc[i+1]
            
            forward_rates[i] = (r2 * t2 - r1 * t1) / (t2 - t1)
        
        return forward_rates
    
    def decompose_yield_curve(self,
                             yields: pd.Series,
                             maturities: np.ndarray) -> Dict:
        """
        Decompose yield curve into level, slope, and curvature.
        
        Args:
            yields: Yield series
            maturities: Maturities
        
        Returns:
            Dictionary with components
        """
        # Level: long-term yield (10Y)
        level = yields.iloc[-1] if len(yields) > 0 else 0
        
        # Slope: difference between long and short term
        if len(yields) >= 2:
            slope = yields.iloc[-1] - yields.iloc[0]
        else:
            slope = 0
        
        # Curvature: 2 * medium - short - long
        if len(yields) >= 3:
            mid_point = len(yields) // 2
            curvature = 2 * yields.iloc[mid_point] - yields.iloc[0] - yields.iloc[-1]
        else:
            curvature = 0
        
        return {
            'level': level,
            'slope': slope,
            'curvature': curvature
        }


class BusinessCycleModel:
    """
    Business cycle analysis and forecasting.
    """
    
    def __init__(self):
        """Initialize business cycle model."""
        pass
    
    def detect_phases(self, gdp_growth: pd.Series) -> pd.Series:
        """
        Detect business cycle phases.
        
        Args:
            gdp_growth: GDP growth rate series
        
        Returns:
            Series with phase labels
        """
        phases = pd.Series('', index=gdp_growth.index, dtype=str)
        
        # Calculate moving average
        ma = gdp_growth.rolling(window=4).mean()
        
        # Expansion: growth above trend
        # Recession: growth below trend and negative
        # Recovery: growth positive but below trend
        
        trend = gdp_growth.mean()
        
        for i in range(len(gdp_growth)):
            if pd.isna(ma.iloc[i]):
                phases.iloc[i] = 'Unknown'
            elif gdp_growth.iloc[i] < 0:
                phases.iloc[i] = 'Recession'
            elif gdp_growth.iloc[i] > trend:
                phases.iloc[i] = 'Expansion'
            else:
                phases.iloc[i] = 'Recovery'
        
        return phases
    
    def calculate_okuns_law(self,
                           gdp_growth: pd.Series,
                           unemployment_change: pd.Series) -> float:
        """
        Calculate Okun's Law coefficient.
        
        Args:
            gdp_growth: GDP growth rate
            unemployment_change: Change in unemployment rate
        
        Returns:
            Okun's coefficient
        """
        # Align indices
        common_index = gdp_growth.index.intersection(unemployment_change.index)
        gdp_aligned = gdp_growth.loc[common_index]
        unemp_aligned = unemployment_change.loc[common_index]
        
        # Remove NaN
        valid = ~(gdp_aligned.isna() | unemp_aligned.isna())
        gdp_clean = gdp_aligned[valid]
        unemp_clean = unemp_aligned[valid]
        
        if len(gdp_clean) < 2:
            return np.nan
        
        # Simple regression: unemployment_change = a + b * gdp_growth
        from sklearn.linear_model import LinearRegression
        
        X = gdp_clean.values.reshape(-1, 1)
        y = unemp_clean.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[0]  # Okun's coefficient


class PhillipsCurveModel:
    """
    Phillips Curve analysis for inflation-unemployment relationship.
    """
    
    def __init__(self):
        """Initialize Phillips Curve model."""
        pass
    
    def estimate_phillips_curve(self,
                                inflation: pd.Series,
                                unemployment: pd.Series,
                                inflation_expectations: Optional[pd.Series] = None) -> Dict:
        """
        Estimate Phillips Curve relationship.
        
        Args:
            inflation: Inflation rate
            unemployment: Unemployment rate
            inflation_expectations: Expected inflation (optional)
        
        Returns:
            Dictionary with model results
        """
        from sklearn.linear_model import LinearRegression
        
        # Align data
        common_index = inflation.index.intersection(unemployment.index)
        inf_aligned = inflation.loc[common_index]
        unemp_aligned = unemployment.loc[common_index]
        
        # Remove NaN
        valid = ~(inf_aligned.isna() | unemp_aligned.isna())
        inf_clean = inf_aligned[valid]
        unemp_clean = unemp_aligned[valid]
        
        if len(inf_clean) < 2:
            return {'error': 'Insufficient data'}
        
        # Model: inflation = a + b * unemployment + c * expected_inflation
        if inflation_expectations is not None:
            exp_aligned = inflation_expectations.loc[common_index]
            exp_clean = exp_aligned[valid]
            
            X = pd.DataFrame({
                'unemployment': unemp_clean,
                'expectations': exp_clean
            }).dropna()
            
            y = inf_clean.loc[X.index]
        else:
            X = pd.DataFrame({'unemployment': unemp_clean})
            y = inf_clean
        
        if len(X) == 0:
            return {'error': 'No valid data after alignment'}
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            'coefficients': pd.Series(model.coef_, index=X.columns),
            'intercept': model.intercept_,
            'r_squared': model.score(X, y),
            'predictions': pd.Series(model.predict(X), index=X.index)
        }
    
    def calculate_nairu(self,
                       inflation: pd.Series,
                       unemployment: pd.Series) -> float:
        """
        Estimate NAIRU (Non-Accelerating Inflation Rate of Unemployment).
        
        Args:
            inflation: Inflation rate
            unemployment: Unemployment rate
        
        Returns:
            Estimated NAIRU
        """
        # Simple approach: NAIRU is unemployment when inflation is stable
        # More sophisticated: use Phillips Curve to find unemployment where inflation change is zero
        
        # Calculate inflation change
        inflation_change = inflation.diff()
        
        # Find periods of stable inflation (change near zero)
        stable_periods = inflation_change.abs() < inflation_change.std() * 0.5
        
        if stable_periods.sum() > 0:
            nairu = unemployment[stable_periods].mean()
        else:
            # Fallback: use median unemployment
            nairu = unemployment.median()
        
        return nairu
