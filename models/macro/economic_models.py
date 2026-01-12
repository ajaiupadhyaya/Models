"""
Macroeconomic Analysis Models
GDP forecasting, yield curve analysis, economic indicators
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class GDPForecaster:
    """
    GDP forecasting model using various methodologies.
    """
    
    def __init__(self, gdp_data: pd.Series):
        """
        Initialize GDP forecaster.
        
        Args:
            gdp_data: Historical GDP time series
        """
        self.gdp_data = gdp_data
        self.model = None
    
    def forecast_linear(self, periods: int = 4) -> pd.Series:
        """
        Forecast GDP using linear trend.
        
        Args:
            periods: Number of periods to forecast
        
        Returns:
            Forecasted GDP series
        """
        X = np.arange(len(self.gdp_data)).reshape(-1, 1)
        y = self.gdp_data.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(self.gdp_data), len(self.gdp_data) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        future_dates = pd.date_range(start=self.gdp_data.index[-1], periods=periods+1, freq='Q')[1:]
        return pd.Series(forecast, index=future_dates)
    
    def forecast_growth_rate(self, 
                            historical_growth: float,
                            periods: int = 4) -> pd.Series:
        """
        Forecast GDP using constant growth rate.
        
        Args:
            historical_growth: Historical average growth rate
            periods: Number of periods to forecast
        
        Returns:
            Forecasted GDP series
        """
        last_value = self.gdp_data.iloc[-1]
        forecast = []
        
        for i in range(1, periods + 1):
            forecast.append(last_value * ((1 + historical_growth) ** i))
        
        future_dates = pd.date_range(start=self.gdp_data.index[-1], periods=periods+1, freq='Q')[1:]
        return pd.Series(forecast, index=future_dates)
    
    def calculate_growth_rate(self) -> float:
        """
        Calculate average GDP growth rate.
        
        Returns:
            Average growth rate
        """
        returns = self.gdp_data.pct_change().dropna()
        return returns.mean()


class YieldCurveAnalyzer:
    """
    Yield curve analysis and forecasting.
    """
    
    def __init__(self, yield_data: pd.DataFrame):
        """
        Initialize yield curve analyzer.
        
        Args:
            yield_data: DataFrame with yields for different maturities
        """
        self.yield_data = yield_data
    
    def calculate_spread(self, short_term: str, long_term: str) -> pd.Series:
        """
        Calculate yield spread between two maturities.
        
        Args:
            short_term: Short-term maturity column name
            long_term: Long-term maturity column name
        
        Returns:
            Spread series
        """
        return self.yield_data[long_term] - self.yield_data[short_term]
    
    def detect_inversion(self, short_term: str = '2Y', long_term: str = '10Y') -> pd.Series:
        """
        Detect yield curve inversion.
        
        Args:
            short_term: Short-term maturity
            long_term: Long-term maturity
        
        Returns:
            Boolean series indicating inversion
        """
        spread = self.calculate_spread(short_term, long_term)
        return spread < 0
    
    def calculate_forward_rates(self) -> pd.DataFrame:
        """
        Calculate implied forward rates from spot rates.
        
        Returns:
            DataFrame with forward rates
        """
        # Simplified calculation - assumes annual compounding
        forward_rates = pd.DataFrame(index=self.yield_data.index)
        
        for i in range(1, len(self.yield_data.columns)):
            col_name = f"F{self.yield_data.columns[i-1]}-{self.yield_data.columns[i]}"
            # Simplified forward rate calculation
            forward_rates[col_name] = (
                (1 + self.yield_data.iloc[:, i]) ** (i+1) / 
                (1 + self.yield_data.iloc[:, i-1]) ** i - 1
            )
        
        return forward_rates


class EconomicIndicatorAnalyzer:
    """
    Analysis of economic indicators.
    """
    
    def __init__(self, indicators: Dict[str, pd.Series]):
        """
        Initialize with economic indicators.
        
        Args:
            indicators: Dictionary of indicator name to series
        """
        self.indicators = indicators
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation between indicators.
        
        Returns:
            Correlation matrix
        """
        df = pd.DataFrame(self.indicators)
        return df.corr()
    
    def detect_recession(self, 
                        gdp_growth: pd.Series,
                        threshold: float = -0.01) -> pd.Series:
        """
        Detect recession periods based on GDP growth.
        
        Args:
            gdp_growth: GDP growth rate series
            threshold: Negative growth threshold
        
        Returns:
            Boolean series indicating recession
        """
        return gdp_growth < threshold
    
    def leading_indicators_index(self, 
                                indicators: List[str],
                                weights: Optional[List[float]] = None) -> pd.Series:
        """
        Create composite leading indicators index.
        
        Args:
            indicators: List of indicator names to include
            weights: Optional weights for each indicator
        
        Returns:
            Composite index series
        """
        if weights is None:
            weights = [1.0 / len(indicators)] * len(indicators)
        
        # Normalize indicators
        normalized = {}
        for ind in indicators:
            if ind in self.indicators:
                series = self.indicators[ind]
                normalized[ind] = (series - series.mean()) / series.std()
        
        # Weighted average
        composite = pd.Series(0, index=list(normalized.values())[0].index)
        for ind, weight in zip(indicators, weights):
            if ind in normalized:
                composite += normalized[ind] * weight
        
        return composite
    
    def forecast_using_indicators(self,
                                 target: str,
                                 predictor_indicators: List[str]) -> Dict:
        """
        Forecast target indicator using other indicators.
        
        Args:
            target: Target indicator name
            predictor_indicators: List of predictor indicator names
        
        Returns:
            Dictionary with forecast results
        """
        if target not in self.indicators:
            raise ValueError(f"Target indicator {target} not found")
        
        # Prepare data
        df = pd.DataFrame({ind: self.indicators[ind] for ind in predictor_indicators 
                          if ind in self.indicators})
        df = df.dropna()
        
        if len(df) == 0:
            return {'error': 'Insufficient data'}
        
        y = self.indicators[target].loc[df.index]
        y = y.dropna()
        common_index = df.index.intersection(y.index)
        
        if len(common_index) == 0:
            return {'error': 'No overlapping data'}
        
        X = df.loc[common_index]
        y = y.loc[common_index]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions
        predictions = model.predict(X)
        
        return {
            'model': model,
            'predictions': pd.Series(predictions, index=common_index),
            'actual': y,
            'r_squared': model.score(X, y),
            'coefficients': pd.Series(model.coef_, index=X.columns)
        }
