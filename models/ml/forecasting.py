"""
Machine Learning Models for Financial Forecasting
Advanced ML techniques for time series prediction and pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.neural_network import MLPRegressor
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False


class TimeSeriesForecaster:
    """
    Advanced time series forecasting using machine learning.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize forecaster.
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.n_lags = 10  # Store n_lags for later use in predict
    
    def create_features(self, series: pd.Series, n_lags: int = 10) -> pd.DataFrame:
        """
        Create features from time series.
        
        Args:
            series: Time series data
            n_lags: Number of lag features
        
        Returns:
            DataFrame with features
        """
        df = pd.DataFrame(index=series.index)
        df['target'] = series
        
        # Lag features
        for lag in range(1, n_lags + 1):
            df[f'lag_{lag}'] = series.shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = series.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = series.rolling(window=window).std()
        
        # Trend features
        df['trend'] = np.arange(len(df))
        df['month'] = df.index.month if hasattr(df.index, 'month') else 0
        df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        
        # Returns
        df['returns'] = series.pct_change()
        df['returns_lag1'] = df['returns'].shift(1)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df.dropna()
    
    def fit(self, series: pd.Series, n_lags: int = 10):
        """
        Fit the forecasting model.
        
        Args:
            series: Time series to fit
            n_lags: Number of lag features
        """
        self.n_lags = n_lags  # Store for use in predict
        # Create features
        feature_df = self.create_features(series, n_lags)
        
        # Prepare data
        X = feature_df.drop('target', axis=1)
        y = feature_df['target']
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'neural_network' and MLP_AVAILABLE:
            self.model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_scaled, y)
    
    def predict(self, series: pd.Series, n_periods: int = 10) -> pd.Series:
        """
        Forecast future values.
        
        Args:
            series: Historical series
            n_periods: Number of periods to forecast
        
        Returns:
            Forecasted series
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecasts = []
        current_series = series.copy()
        
        for _ in range(n_periods):
            # Create features for last point with same n_lags as training
            feature_df = self.create_features(current_series, self.n_lags)
            if len(feature_df) == 0:
                break
            
            X = feature_df.iloc[[-1]].drop('target', axis=1)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            pred = self.model.predict(X_scaled)[0]
            forecasts.append(pred)
            
            # Update series with prediction
            last_date = current_series.index[-1]
            if isinstance(last_date, pd.Timestamp):
                next_date = last_date + pd.Timedelta(days=1)
            else:
                next_date = len(current_series)
            
            current_series = pd.concat([
                current_series,
                pd.Series([pred], index=[next_date])
            ])
        
        # Create forecast index
        last_date = series.index[-1]
        if isinstance(last_date, pd.Timestamp):
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq='D'
            )
        else:
            forecast_index = range(len(series), len(series) + n_periods)
        
        return pd.Series(forecasts, index=forecast_index[:len(forecasts)])
    
    def feature_importance(self) -> pd.Series:
        """
        Get feature importance (for tree-based models).
        
        Returns:
            Series with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        else:
            return pd.Series()


class RegimeDetector:
    """
    Detect market regimes using clustering and statistical methods.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
        """
        self.n_regimes = n_regimes
        self.model = None
    
    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Detect market regimes from returns.
        
        Args:
            returns: Returns series
        
        Returns:
            Series with regime labels
        """
        from sklearn.cluster import KMeans
        
        # Create features
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        rolling_skew = returns.rolling(20).skew()
        
        features = pd.DataFrame({
            'mean': rolling_mean,
            'std': rolling_std,
            'skew': rolling_skew
        }).dropna()
        
        # Cluster
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Map to full series
        regime_series = pd.Series(index=returns.index, dtype=int)
        regime_series.loc[features.index] = labels
        
        return regime_series.fillna(method='ffill')
    
    def get_regime_characteristics(self, returns: pd.Series, 
                                   regimes: pd.Series) -> pd.DataFrame:
        """
        Get characteristics of each regime.
        
        Args:
            returns: Returns series
            regimes: Regime labels
        
        Returns:
            DataFrame with regime characteristics
        """
        results = []
        
        for regime in range(self.n_regimes):
            regime_returns = returns[regimes == regime]
            results.append({
                'regime': regime,
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'count': len(regime_returns)
            })
        
        return pd.DataFrame(results)


class AnomalyDetector:
    """
    Detect anomalies in financial time series.
    """
    
    def __init__(self, method: str = 'isolation_forest'):
        """
        Initialize anomaly detector.
        
        Args:
            method: 'isolation_forest' or 'statistical'
        """
        self.method = method
        self.model = None
    
    def detect(self, series: pd.Series, contamination: float = 0.1) -> pd.Series:
        """
        Detect anomalies.
        
        Args:
            series: Time series
            contamination: Expected proportion of anomalies
        
        Returns:
            Boolean series indicating anomalies
        """
        if self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            # Create features
            features = pd.DataFrame({
                'value': series,
                'returns': series.pct_change(),
                'rolling_mean': series.rolling(20).mean(),
                'rolling_std': series.rolling(20).std()
            }).dropna()
            
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(features)
            
            anomalies = pd.Series(False, index=series.index)
            anomalies.loc[features.index] = predictions == -1
            
            return anomalies
        
        elif self.method == 'statistical':
            # Z-score method
            mean = series.rolling(20).mean()
            std = series.rolling(20).std()
            z_scores = (series - mean) / std
            
            return z_scores.abs() > 3
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
