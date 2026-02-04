"""
Unit tests for models/ml/forecasting.py: TimeSeriesForecaster with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from models.ml.forecasting import TimeSeriesForecaster


@pytest.fixture
def synthetic_series():
    """Synthetic time series: trend + noise (reproducible)."""
    np.random.seed(42)
    n = 150
    trend = np.linspace(100, 110, n)
    noise = np.random.randn(n) * 2
    series = pd.Series(trend + noise, index=pd.date_range("2024-01-01", periods=n, freq="D"))
    return series


def test_time_series_forecaster_fit_predict_shape(synthetic_series):
    """TimeSeriesForecaster fit then predict returns series of length n_periods."""
    forecaster = TimeSeriesForecaster(model_type="random_forest")
    forecaster.fit(synthetic_series, n_lags=5)
    forecast = forecaster.predict(synthetic_series, n_periods=10)
    assert len(forecast) == 10
    assert isinstance(forecast, pd.Series)


def test_time_series_forecaster_no_nan(synthetic_series):
    """Forecast contains no NaN."""
    forecaster = TimeSeriesForecaster(model_type="random_forest")
    forecaster.fit(synthetic_series, n_lags=5)
    forecast = forecaster.predict(synthetic_series, n_periods=5)
    assert not forecast.isna().any()


def test_time_series_forecaster_gradient_boosting(synthetic_series):
    """Gradient boosting model fits and predicts."""
    forecaster = TimeSeriesForecaster(model_type="gradient_boosting")
    forecaster.fit(synthetic_series, n_lags=5)
    forecast = forecaster.predict(synthetic_series, n_periods=5)
    assert len(forecast) == 5
    assert forecaster.model is not None


def test_time_series_forecaster_predict_without_fit_raises(synthetic_series):
    """Predict without fit raises ValueError."""
    forecaster = TimeSeriesForecaster(model_type="random_forest")
    with pytest.raises(ValueError, match="not fitted"):
        forecaster.predict(synthetic_series, n_periods=5)
