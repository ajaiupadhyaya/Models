#!/usr/bin/env python3
"""
Lightweight ML/DL/RL smoke tests to validate pipelines end-to-end.
Runs quickly with minimal epochs and tiny slices of data.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import DataFetcher
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor, RLReadyEnvironment


def _sample_prices(ticker: str = "AAPL", period: str = "6mo"):
    fetcher = DataFetcher()
    df = fetcher.get_stock_data(ticker, period=period)
    if len(df) < 120:
        raise RuntimeError("Insufficient price history for smoke test")
    return df.tail(200)  # keep small for speed


def test_ensemble(df):
    model = EnsemblePredictor(lookback_window=20)
    model.train(df)
    signals = model.predict(df)
    assert len(signals) == len(df)
    print("✓ Ensemble predictor trained and produced signals")


def test_lstm(df):
    try:
        model = LSTMPredictor(lookback_window=20, hidden_units=16)
    except ImportError:
        print("⚠ TensorFlow not available; skipping LSTM smoke test")
        return

    # Reduce epochs for speed
    model.train(df, epochs=2, batch_size=32)
    preds = model.predict(df)
    assert preds.shape[0] == len(df)
    print("✓ LSTM predictor trained and produced predictions")


def test_rl(df):
    env = RLReadyEnvironment(df)
    state = env.reset()
    # take 10 random steps
    rng = np.random.default_rng(42)
    for _ in range(10):
        action = int(rng.integers(0, 4))
        state, reward, done, info = env.step(action)
        if done:
            break
    print("✓ RL environment stepped through actions")


def main():
    df = _sample_prices()
    test_ensemble(df)
    test_lstm(df)
    test_rl(df)
    print("\nSMOKE ML TESTS: PASS")


if __name__ == "__main__":
    main()
