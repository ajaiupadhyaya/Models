"""
Tests for price prediction engine.

Tests:
- Feature engineering (returns, moving averages, RSI, Bollinger Bands, MACD)
- Model training (linear trend, momentum, moving average, ML features)
- Price prediction with confidence intervals
- Model evaluation (MSE, RMSE, MAE, MAPE, R², directional accuracy)
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from core.price_prediction import (
    FeatureEngineer,
    PricePredictor,
    PredictionResult,
    ModelPerformance,
    quick_prediction,
)


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        prices = pd.Series(100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod(), index=dates)
        return prices
    
    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer()
    
    def test_calculate_returns(self, engineer, sample_prices):
        """Test return calculation."""
        returns = engineer.calculate_returns(sample_prices, periods=1)
        
        assert len(returns) == len(sample_prices)
        assert returns.iloc[0] == 0 or np.isnan(returns.iloc[0])  # First return is NaN
        
        # Manual verification
        expected = sample_prices.pct_change(1)
        assert np.allclose(returns[1:], expected[1:], equal_nan=True)
    
    def test_calculate_log_returns(self, engineer, sample_prices):
        """Test log return calculation."""
        log_returns = engineer.calculate_log_returns(sample_prices)
        
        assert len(log_returns) == len(sample_prices)
        # Log returns approximately equal to simple returns for small changes
        simple_returns = engineer.calculate_returns(sample_prices, 1)
        assert np.allclose(log_returns[1:10], simple_returns[1:10], atol=0.01)
    
    def test_calculate_sma(self, engineer, sample_prices):
        """Test simple moving average."""
        sma_20 = engineer.calculate_sma(sample_prices, window=20)
        
        assert len(sma_20) == len(sample_prices)
        # First 19 values should be NaN
        assert sma_20.iloc[:19].isna().all()
        # 20th value should be average of first 20 prices
        assert abs(sma_20.iloc[19] - sample_prices.iloc[:20].mean()) < 0.001
    
    def test_calculate_ema(self, engineer, sample_prices):
        """Test exponential moving average."""
        ema_12 = engineer.calculate_ema(sample_prices, span=12)
        
        assert len(ema_12) == len(sample_prices)
        # EMA should be smoother than price
        assert ema_12.std() < sample_prices.std()
    
    def test_calculate_rsi(self, engineer, sample_prices):
        """Test RSI calculation."""
        rsi = engineer.calculate_rsi(sample_prices, periods=14)
        
        assert len(rsi) == len(sample_prices)
        # RSI should be between 0 and 100
        assert (rsi[20:] >= 0).all()
        assert (rsi[20:] <= 100).all()
    
    def test_calculate_bollinger_bands(self, engineer, sample_prices):
        """Test Bollinger Bands calculation."""
        middle, upper, lower = engineer.calculate_bollinger_bands(sample_prices, window=20)
        
        assert len(middle) == len(sample_prices)
        # Upper should be above middle, lower should be below
        assert (upper[20:] >= middle[20:]).all()
        assert (lower[20:] <= middle[20:]).all()
        # Price should mostly be within bands
        within_bands = ((sample_prices[20:] >= lower[20:]) & (sample_prices[20:] <= upper[20:])).mean()
        assert within_bands > 0.8  # At least 80% within bands
    
    def test_calculate_macd(self, engineer, sample_prices):
        """Test MACD calculation."""
        macd, signal = engineer.calculate_macd(sample_prices)
        
        assert len(macd) == len(sample_prices)
        assert len(signal) == len(sample_prices)
        # MACD and signal should be correlated
        correlation = macd[30:].corr(signal[30:])
        assert correlation > 0.5
    
    def test_create_features(self, engineer, sample_prices):
        """Test comprehensive feature creation."""
        features = engineer.create_features(sample_prices)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_prices)
        
        # Check expected columns
        expected_columns = [
            "price", "returns_1d", "sma_5", "sma_20", "sma_50",
            "ema_12", "ema_26", "rsi", "bb_position", "macd", "volatility_20"
        ]
        for col in expected_columns:
            assert col in features.columns
        
        # No NaN values (should be filled)
        assert not features.isna().any().any()
    
    def test_create_features_with_volume(self, engineer, sample_prices):
        """Test feature creation with volume data."""
        volume = pd.Series(
            np.random.randint(1000000, 10000000, len(sample_prices)),
            index=sample_prices.index
        )
        
        features = engineer.create_features(sample_prices, volume)
        
        # Volume features should be present
        assert "volume" in features.columns
        assert "volume_sma_20" in features.columns
        assert "volume_ratio" in features.columns


class TestPricePredictor:
    """Test price prediction models."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series with trend."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        # Trending upward with noise
        trend = np.linspace(100, 120, 100)
        noise = np.random.normal(0, 1, 100)
        prices = pd.Series(trend + noise, index=dates)
        return prices
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = PricePredictor(model_type="linear_trend", lookback_window=30)
        
        assert predictor.model_type == "linear_trend"
        assert predictor.lookback_window == 30
        assert not predictor.trained
    
    def test_train_linear_trend(self, sample_prices):
        """Test training with linear trend model."""
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(sample_prices)
        
        assert predictor.trained
        assert predictor.model is not None
        assert len(predictor.feature_names) > 0
    
    def test_train_momentum(self, sample_prices):
        """Test training with momentum model."""
        predictor = PricePredictor(model_type="momentum")
        predictor.train(sample_prices)
        
        assert predictor.trained
        assert "returns_5d" in predictor.feature_names or "momentum" in str(predictor.feature_names)
    
    def test_train_moving_average(self, sample_prices):
        """Test training with moving average model."""
        predictor = PricePredictor(model_type="moving_average")
        predictor.train(sample_prices)
        
        assert predictor.trained
        assert "sma" in str(predictor.feature_names).lower()
    
    def test_train_ml_features(self, sample_prices):
        """Test training with full ML features."""
        predictor = PricePredictor(model_type="ml_features")
        predictor.train(sample_prices)
        
        assert predictor.trained
        # Should have multiple features
        assert len(predictor.feature_names) > 3
    
    def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        prices = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))
        
        predictor = PricePredictor()
        
        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.train(prices)
    
    def test_predict(self, sample_prices):
        """Test price prediction."""
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(sample_prices)
        
        result = predictor.predict(sample_prices, horizon=5)
        
        assert isinstance(result, PredictionResult)
        assert len(result.predicted_prices) == 5
        assert len(result.prediction_dates) == 5
        assert len(result.confidence_intervals) == 5
        assert result.model_type == "linear_trend"
        assert result.current_price == sample_prices.iloc[-1]
    
    def test_predict_without_training(self, sample_prices):
        """Test prediction without training raises error."""
        predictor = PricePredictor()
        
        with pytest.raises(ValueError, match="not trained"):
            predictor.predict(sample_prices)
    
    def test_prediction_trend(self, sample_prices):
        """Test that predictions follow trend."""
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(sample_prices)
        
        result = predictor.predict(sample_prices, horizon=5)
        
        # With upward trend, predictions should generally increase
        predictions = result.predicted_prices
        # At least some predictions should be higher than current
        assert any(p > result.current_price for p in predictions)
    
    def test_confidence_intervals(self, sample_prices):
        """Test confidence intervals."""
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(sample_prices)
        
        result = predictor.predict(sample_prices, horizon=5, confidence_level=0.95)
        
        # Confidence intervals should contain predictions
        for pred, (lower, upper) in zip(result.predicted_prices, result.confidence_intervals):
            assert lower <= pred <= upper
        
        # Intervals should widen with horizon
        widths = [upper - lower for lower, upper in result.confidence_intervals]
        assert widths[-1] >= widths[0]  # Later intervals should be wider
    
    def test_evaluate(self, sample_prices):
        """Test model evaluation."""
        predictor = PricePredictor(model_type="linear_trend", lookback_window=60)
        predictor.train(sample_prices)
        
        performance = predictor.evaluate(sample_prices, test_size=20)
        
        assert isinstance(performance, ModelPerformance)
        assert performance.model_type == "linear_trend"
        
        # Check metrics are reasonable
        assert performance.mse >= 0
        assert performance.rmse >= 0
        assert performance.mae >= 0
        assert performance.mape >= 0
        assert 0 <= performance.directional_accuracy <= 100
        # R² can be very negative for poor models (worse than mean baseline)
        assert performance.r2_score <= 1  # Just check upper bound
        assert np.isfinite(performance.r2_score)
    
    def test_prediction_result_to_dict(self, sample_prices):
        """Test PredictionResult serialization."""
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(sample_prices)
        result = predictor.predict(sample_prices, horizon=3)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "predicted_prices" in result_dict
        assert "confidence_intervals" in result_dict
        assert "model_type" in result_dict
        assert len(result_dict["predicted_prices"]) == 3
    
    def test_model_performance_to_dict(self, sample_prices):
        """Test ModelPerformance serialization."""
        predictor = PricePredictor(model_type="linear_trend", lookback_window=60)
        predictor.train(sample_prices)
        performance = predictor.evaluate(sample_prices, test_size=20)
        
        perf_dict = performance.to_dict()
        
        assert isinstance(perf_dict, dict)
        assert "mse" in perf_dict
        assert "rmse" in perf_dict
        assert "mae" in perf_dict
        assert "directional_accuracy" in perf_dict


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_quick_prediction(self):
        """Test quick prediction function."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        prices = pd.Series(100 + np.arange(50) * 0.5, index=dates)  # Linear trend
        
        predictions = quick_prediction(prices, horizon=3)
        
        assert len(predictions) == 3
        assert all(isinstance(p, float) for p in predictions)
        # With upward trend, predictions should increase
        assert predictions[-1] > predictions[0]


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def realistic_prices(self):
        """Generate realistic stock price data."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        np.random.seed(123)
        # Simulate realistic stock with trend and volatility
        returns = np.random.normal(0.0005, 0.015, 252)
        prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
        return prices
    
    def test_full_prediction_pipeline(self, realistic_prices):
        """Test complete prediction workflow."""
        # Train model
        predictor = PricePredictor(model_type="ml_features", lookback_window=180)
        predictor.train(realistic_prices)
        
        # Make prediction
        result = predictor.predict(realistic_prices, horizon=10)
        
        # Evaluate
        performance = predictor.evaluate(realistic_prices, test_size=40)
        
        # Verify results
        assert result.current_price > 0
        assert len(result.predicted_prices) == 10
        assert performance.mse > 0
        assert 0 <= performance.directional_accuracy <= 100
    
    def test_compare_models(self, realistic_prices):
        """Test comparing different models."""
        models = ["linear_trend", "momentum", "moving_average", "ml_features"]
        performances = []
        
        for model_type in models:
            predictor = PricePredictor(model_type=model_type, lookback_window=120)
            predictor.train(realistic_prices)
            perf = predictor.evaluate(realistic_prices, test_size=30)
            performances.append((model_type, perf.mae))
        
        # All models should complete without errors
        assert len(performances) == 4
        
        # All MAEs should be positive and finite
        for model_type, mae in performances:
            assert mae > 0
            assert np.isfinite(mae)
    
    def test_feature_engineering_integration(self, realistic_prices):
        """Test feature engineering with prediction."""
        engineer = FeatureEngineer()
        features = engineer.create_features(realistic_prices)
        
        # Train predictor with engineered features
        predictor = PricePredictor(model_type="ml_features")
        predictor.train(realistic_prices)
        
        # Predict
        result = predictor.predict(realistic_prices, horizon=5)
        
        assert len(result.predicted_prices) == 5
        # Features should have been used
        assert len(predictor.feature_names) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_flat_prices(self):
        """Test with completely flat prices."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        prices = pd.Series([100] * 50, index=dates)
        
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(prices)
        result = predictor.predict(prices, horizon=3)
        
        # Should predict flat continuation
        assert all(abs(p - 100) < 5 for p in result.predicted_prices)
    
    def test_extreme_volatility(self):
        """Test with extremely volatile prices."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        prices = pd.Series(100 * (1 + np.random.normal(0, 0.1, 50)).cumprod(), index=dates)
        
        predictor = PricePredictor(model_type="linear_trend")
        predictor.train(prices)
        result = predictor.predict(prices, horizon=3)
        
        # Should still make predictions
        assert len(result.predicted_prices) == 3
        # Confidence intervals should be wide
        widths = [upper - lower for lower, upper in result.confidence_intervals]
        assert all(w > 0 for w in widths)
    
    def test_short_horizon(self):
        """Test with horizon=1."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        prices = pd.Series(100 + np.arange(50), index=dates)
        
        predictor = PricePredictor()
        predictor.train(prices)
        result = predictor.predict(prices, horizon=1)
        
        assert len(result.predicted_prices) == 1
        assert len(result.confidence_intervals) == 1
    
    def test_long_horizon(self):
        """Test with long prediction horizon."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = pd.Series(100 + np.arange(100) * 0.5, index=dates)
        
        predictor = PricePredictor()
        predictor.train(prices)
        result = predictor.predict(prices, horizon=30)
        
        assert len(result.predicted_prices) == 30
        # Confidence should decrease with longer horizon (wider intervals)
        widths = [upper - lower for lower, upper in result.confidence_intervals]
        assert widths[-1] > widths[0]
