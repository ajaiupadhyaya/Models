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

try:
    from .feature_engineering import (
        LabelGenerator,
        FeatureTransformer
    )
except ImportError:
    LabelGenerator = None
    FeatureTransformer = None

__all__ = [
    'TimeSeriesForecaster',
    'AnomalyDetector',
    'LSTMPredictor',
    'EnsemblePredictor',
    'RLReadyEnvironment',
    'LabelGenerator',
    'FeatureTransformer'
]
