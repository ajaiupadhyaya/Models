"""
Price Prediction Engine

Machine learning models for stock price forecasting:
- Time series forecasting (ARIMA, exponential smoothing)
- Feature engineering (technical indicators, momentum, volatility)
- Model training and prediction
- Backtesting and performance evaluation
- Ensemble methods for improved accuracy

Usage:
    from core.price_prediction import PricePredictor, PredictionResult
    
    predictor = PricePredictor(model_type="linear_trend")
    predictor.train(historical_prices)
    prediction = predictor.predict(horizon=5)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Price prediction result."""
    
    symbol: str
    current_price: float
    predicted_prices: List[float]  # Predictions for next N days
    prediction_dates: List[datetime]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper) bounds
    model_type: str
    features_used: List[str]
    trained_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "predicted_prices": self.predicted_prices,
            "prediction_dates": [d.isoformat() for d in self.prediction_dates],
            "confidence_intervals": self.confidence_intervals,
            "model_type": self.model_type,
            "features_used": self.features_used,
            "trained_at": self.trained_at.isoformat(),
        }


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    
    model_type: str
    mse: float  # Mean squared error
    rmse: float  # Root mean squared error
    mae: float  # Mean absolute error
    mape: float  # Mean absolute percentage error
    r2_score: float  # R-squared
    directional_accuracy: float  # % of correct up/down predictions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2_score": self.r2_score,
            "directional_accuracy": self.directional_accuracy,
        }


class FeatureEngineer:
    """
    Feature engineering for price prediction.
    
    Creates technical indicators and derived features from price data.
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate percentage returns."""
        return prices.pct_change(periods=periods)
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns."""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Calculate simple moving average."""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
        """Calculate exponential moving average."""
        return prices.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands (middle, upper, lower)."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return middle, upper, lower
    
    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, signal_line
    
    def create_features(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            prices: Price series
            volume: Optional volume series
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=prices.index)
        
        # Price-based features
        features["price"] = prices
        features["returns_1d"] = self.calculate_returns(prices, 1)
        features["returns_5d"] = self.calculate_returns(prices, 5)
        features["log_returns"] = self.calculate_log_returns(prices)
        
        # Moving averages
        features["sma_5"] = self.calculate_sma(prices, 5)
        features["sma_20"] = self.calculate_sma(prices, 20)
        features["sma_50"] = self.calculate_sma(prices, 50)
        features["ema_12"] = self.calculate_ema(prices, 12)
        features["ema_26"] = self.calculate_ema(prices, 26)
        
        # Technical indicators
        features["rsi"] = self.calculate_rsi(prices)
        
        bb_middle, bb_upper, bb_lower = self.calculate_bollinger_bands(prices)
        features["bb_position"] = (prices - bb_lower) / (bb_upper - bb_lower)
        
        macd, signal = self.calculate_macd(prices)
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_diff"] = macd - signal
        
        # Volatility
        features["volatility_20"] = prices.pct_change().rolling(20).std()
        
        # Momentum
        features["momentum_5"] = prices / prices.shift(5) - 1
        features["momentum_20"] = prices / prices.shift(20) - 1
        
        # Volume features (if available)
        if volume is not None:
            features["volume"] = volume
            features["volume_sma_20"] = volume.rolling(20).mean()
            features["volume_ratio"] = volume / volume.rolling(20).mean()
        
        return features.bfill().fillna(0)


class PricePredictor:
    """
    Price prediction using machine learning.
    
    Models:
    - linear_trend: Linear regression on time
    - momentum: Momentum-based prediction
    - moving_average: Moving average crossover
    - ml_features: ML model with engineered features
    """
    
    def __init__(
        self,
        model_type: str = "linear_trend",
        lookback_window: int = 30,
    ):
        """
        Initialize predictor.
        
        Args:
            model_type: Model type to use
            lookback_window: Number of days to use for training
        """
        self.model_type = model_type
        self.lookback_window = lookback_window
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.trained = False
        self.feature_names = []
    
    def prepare_data(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data.
        
        Args:
            prices: Price series
            volume: Optional volume series
            
        Returns:
            Tuple of (X, y) for training
        """
        if self.model_type == "linear_trend":
            # Simple time-based features
            X = np.arange(len(prices)).reshape(-1, 1)
            y = prices.values
            self.feature_names = ["time"]
        
        elif self.model_type == "momentum":
            # Momentum-based features
            returns = prices.pct_change(5).values
            X = returns[:-1].reshape(-1, 1)
            y = prices.values[1:]
            self.feature_names = ["returns_5d"]
        
        elif self.model_type == "moving_average":
            # Moving average features
            sma_5 = prices.rolling(5).mean().values
            sma_20 = prices.rolling(20).mean().values
            X = np.column_stack([sma_5[:-1], sma_20[:-1]])
            y = prices.values[1:]
            self.feature_names = ["sma_5", "sma_20"]
        
        else:  # ml_features
            # Full feature set
            features_df = self.feature_engineer.create_features(prices, volume)
            
            # Select relevant features (exclude price itself and NaN-heavy features)
            feature_cols = [
                "returns_1d", "returns_5d", "sma_5", "sma_20", "sma_50",
                "ema_12", "ema_26", "rsi", "bb_position", "macd_diff",
                "volatility_20", "momentum_5", "momentum_20",
            ]
            
            if volume is not None:
                feature_cols.extend(["volume_ratio"])
            
            X = features_df[feature_cols].values[:-1]
            y = prices.values[1:]
            self.feature_names = feature_cols
        
        # Remove any remaining NaNs
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ):
        """
        Train the prediction model.
        
        Args:
            prices: Historical prices
            volume: Optional volume data
        """
        # Take last N days according to lookback window
        if len(prices) > self.lookback_window:
            prices = prices.iloc[-self.lookback_window:]
            if volume is not None:
                volume = volume.iloc[-self.lookback_window:]
        
        # Prepare data
        X, y = self.prepare_data(prices, volume)
        
        if len(X) < 10:
            raise ValueError("Insufficient data for training (need at least 10 samples)")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.model_type in ["linear_trend", "momentum", "moving_average"]:
            self.model = LinearRegression()
        else:
            self.model = Ridge(alpha=1.0)  # Regularized for feature-rich model
        
        self.model.fit(X_scaled, y)
        self.trained = True
        
        logger.info(f"Trained {self.model_type} model on {len(X)} samples")
    
    def predict(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        horizon: int = 5,
        confidence_level: float = 0.95,
    ) -> PredictionResult:
        """
        Make price predictions.
        
        Args:
            prices: Recent price history
            volume: Optional volume data
            horizon: Number of days to predict
            confidence_level: Confidence level for intervals
            
        Returns:
            PredictionResult object
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        confidence_intervals = []
        
        # Get current price
        current_price = prices.iloc[-1]
        
        # Calculate prediction error std for confidence intervals
        X, y = self.prepare_data(prices, volume)
        X_scaled = self.scaler.transform(X)
        y_pred_train = self.model.predict(X_scaled)
        residuals = y - y_pred_train
        std_error = np.std(residuals)
        
        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Make predictions
        for i in range(horizon):
            if self.model_type == "linear_trend":
                # Linear extrapolation
                X_future = np.array([[len(prices) + i]])
                X_future_scaled = self.scaler.transform(X_future)
                pred = self.model.predict(X_future_scaled)[0]
            
            elif self.model_type == "momentum":
                # Use last known momentum
                last_return = prices.pct_change(5).iloc[-1]
                X_future = np.array([[last_return]])
                X_future_scaled = self.scaler.transform(X_future)
                pred = self.model.predict(X_future_scaled)[0]
            
            elif self.model_type == "moving_average":
                # Use current moving averages
                sma_5 = prices.rolling(5).mean().iloc[-1]
                sma_20 = prices.rolling(20).mean().iloc[-1]
                X_future = np.array([[sma_5, sma_20]])
                X_future_scaled = self.scaler.transform(X_future)
                pred = self.model.predict(X_future_scaled)[0]
            
            else:  # ml_features
                # Use latest features
                features_df = self.feature_engineer.create_features(prices, volume)
                X_future = features_df[self.feature_names].iloc[-1].values.reshape(1, -1)
                X_future_scaled = self.scaler.transform(X_future)
                pred = self.model.predict(X_future_scaled)[0]
            
            predictions.append(float(pred))
            
            # Confidence interval (expands with horizon)
            margin = z_score * std_error * np.sqrt(i + 1)
            confidence_intervals.append((float(pred - margin), float(pred + margin)))
        
        # Generate prediction dates
        last_date = prices.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        return PredictionResult(
            symbol="UNKNOWN",  # To be filled by caller
            current_price=float(current_price),
            predicted_prices=predictions,
            prediction_dates=prediction_dates,
            confidence_intervals=confidence_intervals,
            model_type=self.model_type,
            features_used=self.feature_names,
            trained_at=datetime.now(),
        )
    
    def evaluate(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        test_size: int = 20,
    ) -> ModelPerformance:
        """
        Evaluate model performance on test data.
        
        Args:
            prices: Price series
            volume: Optional volume
            test_size: Number of samples to use for testing
            
        Returns:
            ModelPerformance metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Split data
        train_prices = prices.iloc[:-test_size]
        test_prices = prices.iloc[-test_size:]
        
        train_volume = volume.iloc[:-test_size] if volume is not None else None
        test_volume = volume.iloc[-test_size:] if volume is not None else None
        
        # Retrain on train split
        self.train(train_prices, train_volume)
        
        # Prepare test data
        X_test, y_test = self.prepare_data(test_prices, test_volume)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return ModelPerformance(
            model_type=self.model_type,
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            r2_score=float(r2),
            directional_accuracy=float(directional_accuracy),
        )


# Convenience functions
def quick_prediction(prices: pd.Series, horizon: int = 5) -> List[float]:
    """
    Quick price prediction using linear trend.
    
    Args:
        prices: Historical prices
        horizon: Days to predict
        
    Returns:
        List of predicted prices
    """
    predictor = PricePredictor(model_type="linear_trend")
    predictor.train(prices)
    result = predictor.predict(prices, horizon=horizon)
    return result.predicted_prices
