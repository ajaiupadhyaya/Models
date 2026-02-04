"""
Unit tests for models/ml/advanced_trading.py: EnsemblePredictor (and LSTMPredictor if available).
Synthetic OHLCV -> fit/predict; assert signal shape and value range.
"""

import numpy as np
import pandas as pd
import pytest

from models.ml.advanced_trading import EnsemblePredictor


@pytest.fixture
def synthetic_ohlcv():
    """Synthetic OHLCV DataFrame (reproducible)."""
    np.random.seed(42)
    n = 120
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 95)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = np.random.randint(500_000, 2_000_000, n)
    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_ensemble_predictor_fit_predict_shape(synthetic_ohlcv):
    """EnsemblePredictor train then predict returns array of same length as df."""
    model = EnsemblePredictor(lookback_window=20)
    model.train(synthetic_ohlcv)
    signals = model.predict(synthetic_ohlcv)
    assert len(signals) == len(synthetic_ohlcv)
    assert isinstance(signals, np.ndarray)


def test_ensemble_predictor_signals_in_range(synthetic_ohlcv):
    """Signals are in [-1, 1] (or close due to clip)."""
    model = EnsemblePredictor(lookback_window=20)
    model.train(synthetic_ohlcv)
    signals = model.predict(synthetic_ohlcv)
    assert signals.min() >= -1.01
    assert signals.max() <= 1.01


def test_ensemble_predictor_predict_without_train_raises(synthetic_ohlcv):
    """Predict without train raises ValueError."""
    model = EnsemblePredictor(lookback_window=20)
    with pytest.raises(ValueError, match="trained"):
        model.predict(synthetic_ohlcv)


def test_rl_environment_steps():
    """RLReadyEnvironment can step through actions (no crash)."""
    from models.ml.advanced_trading import RLReadyEnvironment
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "Close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "Volume": np.full(n, 1_000_000),
        "High": 101 + np.random.rand(n),
        "Low": 99 - np.random.rand(n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    env = RLReadyEnvironment(df)
    state = env.reset()
    for _ in range(5):
        state, reward, done, info = env.step(0)  # action 0
        if done:
            break
    assert True  # no crash
