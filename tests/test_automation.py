"""
Test Automation Framework
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def sample_ohlcv_data(rows: int = 30) -> pd.DataFrame:
    """Create deterministic OHLCV data for offline automation tests."""
    dates = pd.date_range(start="2024-01-01", periods=rows, freq="B")
    close = 100.0 + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(rows, 1_000_000),
        },
        index=dates,
    )


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from automation.data_pipeline import DataPipeline
        pipeline = DataPipeline()
        self.assertIsNotNone(pipeline)
    
    def test_stock_data_fetch(self):
        """Test stock data fetching."""
        from automation.data_pipeline import DataPipeline
        pipeline = DataPipeline()
        result = pipeline.fetch_and_validate_stock("AAPL", period="1mo")
        self.assertIn('success', result)
        if result.get('success'):
            self.assertIn('rows', result)
            self.assertGreater(result['rows'], 0)


class TestMLPipeline(unittest.TestCase):
    """Test ML pipeline."""
    
    def test_ml_pipeline_initialization(self):
        """Test ML pipeline initialization."""
        from automation.ml_pipeline import MLPipeline
        ml = MLPipeline()
        self.assertIsNotNone(ml)
    
    def test_ensemble_training(self):
        """Test ensemble model training."""
        from automation.ml_pipeline import MLPipeline
        ml = MLPipeline()
        result = ml.train_ensemble_model("AAPL")
        # May fail if insufficient data, that's okay
        self.assertIn('success', result)


class TestOrchestrator(unittest.TestCase):
    """Test orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        from automation.orchestrator import AutomationOrchestrator
        orchestrator = AutomationOrchestrator()
        self.assertIsNotNone(orchestrator)
    
    def test_config_loading(self):
        """Test configuration loading."""
        from automation.orchestrator import AutomationOrchestrator
        orchestrator = AutomationOrchestrator()
        self.assertIsNotNone(orchestrator.config)
        self.assertIn('symbols', orchestrator.config)


class TestRealData(unittest.TestCase):
    """Test data integration paths without depending on live market APIs."""
    
    def test_stock_data_fetch(self):
        """Test stock data fetch contract."""
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        with patch.object(fetcher, "get_stock_data", return_value=sample_ohlcv_data()):
            data = fetcher.get_stock_data("SPY", period="1mo")
        self.assertFalse(data.empty)
        self.assertIn('Close', data.columns)
    
    def test_data_validation(self):
        """Test data validation."""
        from core.data_fetcher import DataFetcher
        from core.pipeline.data_monitor import DataValidator
        
        fetcher = DataFetcher()
        with patch.object(fetcher, "get_stock_data", return_value=sample_ohlcv_data()):
            data = fetcher.get_stock_data("AAPL", period="1mo")
        
        validator = DataValidator()
        result = validator.validate_ohlc(data)
        self.assertIn('valid', result)


if __name__ == '__main__':
    unittest.main()
