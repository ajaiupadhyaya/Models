"""
Advanced time-series models using pmdarima and tsfresh.
Provides auto-ARIMA, seasonal ARIMA, and automatic feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, ComprehensiveFCParameters
import warnings
warnings.filterwarnings('ignore')


class AutoArimaForecaster:
    """
    Auto-ARIMA forecaster using pmdarima.
    Automatically detects ARIMA(p,d,q) and SARIMAX(p,d,q)(P,D,Q,m) parameters.
    """
    
    def __init__(self, seasonal: bool = True, m: int = 252, verbose: int = 0):
        """
        Initialize auto-ARIMA forecaster.
        
        Args:
            seasonal: Include seasonality in model
            m: Seasonal period (252 for daily equity data = 1 year)
            verbose: Output verbosity level
        """
        self.seasonal = seasonal
        self.m = m
        self.verbose = verbose
        self.model = None
        self.model_order = None
        self.seasonal_order = None
    
    def fit(self, series: pd.Series, max_p: int = 5, max_q: int = 5) -> Dict:
        """
        Fit auto-ARIMA model to time-series.
        
        Args:
            series: Time-series data
            max_p: Maximum p order
            max_q: Maximum q order
        
        Returns:
            Dictionary with model info and fit statistics
        """
        # Determine differencing order
        d = ndiffs(series, alpha=0.05, max_d=2, test='kpss')
        
        # Fit auto-ARIMA
        self.model = auto_arima(
            series,
            start_p=0,
            start_q=0,
            max_p=max_p,
            max_d=2,
            max_q=max_q,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            trace=self.verbose > 0,
            error_action='ignore',
            suppress_warnings=True,
            maxiter=200,
            return_valid_fits=False
        )
        
        self.model_order = self.model.order
        self.seasonal_order = self.model.seasonal_order if self.seasonal else None
    
        # Handle edge cases where seasonal is too aggressively applied
        if self.model is None or self.model.order is None:
            raise ValueError("Auto-ARIMA failed to fit. Try with seasonal=False or longer data series.")
        
        return {
            'order': self.model.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.model.aic(),
            'bic': self.model.bic(),
            'fit_summary': str(self.model.summary())
        }
    
    def forecast(self, steps: int = 20, confidence: float = 0.95) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate forecast and confidence intervals.
        
        Args:
            steps: Number of steps ahead to forecast
            confidence: Confidence level (0.95 for 95% CI)
        
        Returns:
            Tuple of (forecast_series, confidence_intervals_df)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            forecast, conf_int = self.model.predict(
                n_periods=steps,
                return_conf_int=True,
                alpha=1 - confidence
            )
            conf_int = pd.DataFrame(conf_int, columns=['lower', 'upper'])
        except Exception:
            forecast = self.model.predict(n_periods=steps)
            residuals = self.model.resid()
            sigma = np.sqrt(np.sum(residuals**2) / max(len(residuals) - 1, 1))
            z_score = 1.96 if confidence == 0.95 else 1.645
            margin_of_error = z_score * sigma
            conf_int = pd.DataFrame({
                'lower': forecast - margin_of_error,
                'upper': forecast + margin_of_error
            }, index=range(1, steps + 1))

        return pd.Series(forecast), conf_int
    
    def forecast_with_errors(self, series: pd.Series, steps: int = 20) -> Dict:
        """
        Fit model and produce forecast with error metrics.
        
        Args:
            series: Training series
            steps: Forecast steps
        
        Returns:
            Dict with forecast, CI, and model diagnostics
        """
        fit_info = self.fit(series)
        forecast, conf_int = self.forecast(steps)
        
        return {
            'forecast': forecast,
            'conf_int': conf_int,
            'model_order': fit_info['order'],
            'aic': fit_info['aic'],
            'bic': fit_info['bic'],
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        }


class SeasonalDecomposer:
    """Decompose time-series into trend, seasonal, residual."""
    
    @staticmethod
    def decompose(series: pd.Series, period: int = 252) -> Dict:
        """
        Seasonal decomposition (additive).
        
        Args:
            series: Time-series
            period: Seasonal period
        
        Returns:
            Dict with trend, seasonal, residual, remainder
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }


class TSFeatureExtractor:
    """
    Automatic time-series feature extraction using tsfresh.
    Extracts 700+ features for ML models.
    """
    
    @staticmethod
    def extract_relevant_features(
        df: pd.DataFrame,
        column: str,
        kind: str = 'minimal',
        max_features: int = 50
    ) -> pd.DataFrame:
        """
        Extract relevant features (filter out non-informative ones).
        
        Args:
            df: DataFrame with time-series data
            column: Column name containing time-series
            kind: 'minimal' (~25 features) or 'comprehensive' (700+ features)
            max_features: Max number of features to return
        
        Returns:
            DataFrame with extracted features
        """
        # Prepare data for tsfresh
        ts_data = df[[column]].reset_index(drop=True)
        ts_data['id'] = 1  # Single time-series ID
        ts_data['time'] = range(len(ts_data))
        
        if kind == 'minimal':
            fc_params = MinimalFCParameters()
            features = extract_features(
                ts_data,
                column_id='id',
                column_sort='time',
                default_fc_parameters=fc_params,
                n_jobs=0,
                show_warnings=False
            )
        else:
            fc_params = ComprehensiveFCParameters()
            features = extract_features(
                ts_data,
                column_id='id',
                column_sort='time',
                default_fc_parameters=fc_params,
                parallelization='multiprocessing',
                n_jobs=-1,
                show_warnings=False
            )
        
        return features.iloc[:, :max_features] if len(features.columns) > max_features else features
    
    @staticmethod
    def extract_multiple_columns(
        df: pd.DataFrame,
        kind: str = 'minimal'
    ) -> pd.DataFrame:
        """
        Extract features from multiple columns.
        
        Args:
            df: DataFrame with multiple time-series
            kind: 'minimal' or 'comprehensive'
        
        Returns:
            DataFrame with feature matrix
        """
        all_features = {}
        
        for col in df.columns:
            ts_data = df[[col]].reset_index(drop=True)
            ts_data['id'] = col
            ts_data['time'] = range(len(ts_data))
            
            if kind == 'minimal':
                features = extract_relevant_features(
                    ts_data,
                    target=None,
                    column_id='id',
                    column_sort='time',
                    kind=kind,
                    n_jobs=0,
                    show_warnings=False
                )
            else:
                features = extract_features(
                    ts_data,
                    column_id='id',
                    column_sort='time',
                    n_jobs=0,
                    show_warnings=False
                )
            
            all_features[col] = features
        
        # Concatenate with column names as prefix
        result = pd.concat(
            [features.add_prefix(f'{col}_') for col, features in all_features.items()],
            axis=1
        )
        
        return result


class UnivariateForecaster:
    """Wrapper for simple univariate forecasting."""
    
    def __init__(self, series: pd.Series, seasonal: bool = True):
        self.series = series
        self.forecaster = AutoArimaForecaster(seasonal=seasonal)
        self.fit_info = None
    
    def fit_and_forecast(self, steps: int = 20) -> Dict:
        """Convenience method: fit and forecast in one call."""
        self.fit_info = self.forecaster.fit(self.series)
        forecast, conf_int = self.forecaster.forecast(steps=steps)
        
        return {
            'forecast': forecast,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1],
            'aic': self.fit_info['aic'],
            'model_order': self.fit_info['order']
        }
