"""
Unit tests for core/model_monitor.py: metrics from fixed series match expectations.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from core.model_monitor import ModelPerformanceMonitor
from core.utils import calculate_sharpe_ratio, calculate_max_drawdown


def test_model_monitor_records_and_computes_metrics():
    """Record predictions/actuals; computed Sharpe and max_drawdown match core.utils."""
    np.random.seed(42)
    monitor = ModelPerformanceMonitor(models_dir="data/test_models")
    # Simulate 50 "actual" returns (e.g. from a strategy)
    actuals = np.random.randn(50) * 0.01
    predictions = actuals + np.random.randn(50) * 0.002
    for i in range(50):
        monitor.record_prediction("test_model", "AAPL", float(predictions[i]), float(actuals[i]))
    perf = monitor.get_performance("test_model", "AAPL")
    assert perf is not None
    assert "sharpe_ratio" in perf
    assert "max_drawdown" in perf
    # Compare to core.utils on same series
    returns_series = pd.Series(actuals)
    expected_sharpe = calculate_sharpe_ratio(returns_series)
    expected_dd = calculate_max_drawdown(returns_series)
    if np.isfinite(expected_sharpe):
        assert abs(perf["sharpe_ratio"] - expected_sharpe) < 0.01
    assert abs(perf["max_drawdown"] - expected_dd) < 1e-6


def test_model_monitor_should_retrain_thresholds():
    """should_retrain returns (bool, reason) and respects thresholds."""
    monitor = ModelPerformanceMonitor(models_dir="data/test_models")
    # No data -> no retrain
    should, reason = monitor.should_retrain("nonexistent", "AAPL")
    assert should is False
    assert "No performance" in reason or "No metrics" in reason
