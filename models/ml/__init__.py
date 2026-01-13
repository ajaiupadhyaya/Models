"""
ML module for trading and predictions
"""

try:
    from .forecasting import (
        TimeSeriesForecaster,
        AnomalyDetector
    )
except ImportError:
    TimeSeriesForecaster = None
    AnomalyDetector = None

try:
    from .advanced_trading import (
        LSTMPredictor,
        EnsemblePredictor,
        RLReadyEnvironment
    )
except ImportError:
    LSTMPredictor = None
    EnsemblePredictor = None
    RLReadyEnvironment = None

__all__ = [
    'TimeSeriesForecaster',
    'AnomalyDetector',
    'LSTMPredictor',
    'EnsemblePredictor',
    'RLReadyEnvironment'
]
