"""
Smoke tests for ML/DL/RL pipelines using synthetic data (no network).
Can be run in CI without API keys or live data.
"""

import numpy as np
import pandas as pd
import pytest

from models.ml.advanced_trading import EnsemblePredictor, RLReadyEnvironment


def _synthetic_ohlcv(n=200):
    """Synthetic OHLCV for smoke tests."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 90)
    return pd.DataFrame({
        "Open": np.roll(close, 1),
        "High": close + np.abs(np.random.randn(n)),
        "Low": close - np.abs(np.random.randn(n)),
        "Close": close,
        "Volume": np.random.randint(500_000, 2_000_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_smoke_ensemble():
    """Ensemble predictor: train and predict on synthetic data (smoke)."""
    df = _synthetic_ohlcv(120)
    model = EnsemblePredictor(lookback_window=20)
    model.train(df)
    signals = model.predict(df)
    assert len(signals) == len(df)
    assert not np.any(np.isnan(signals))


def test_smoke_lstm():
    """LSTM predictor: train and predict if TensorFlow available (smoke)."""
    pytest.importorskip("tensorflow", reason="TensorFlow not installed")
    from models.ml.advanced_trading import LSTMPredictor
    df = _synthetic_ohlcv(80)
    model = LSTMPredictor(lookback_window=20, hidden_units=16)
    model.train(df, epochs=2, batch_size=32)
    preds = model.predict(df)
    assert preds.shape[0] == len(df)


def test_smoke_rl():
    """RL environment: reset and step (smoke)."""
    df = _synthetic_ohlcv(60)
    env = RLReadyEnvironment(df)
    state = env.reset()
    for _ in range(10):
        state, reward, done, info = env.step(np.random.randint(0, 4))
        if done:
            break
    assert True
