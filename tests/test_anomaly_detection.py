"""
Tests for anomaly detection engine.

Tests:
- Statistical anomaly detection (z-score, IQR, modified z-score)
- Time series anomalies (level shifts, trend changes)
- ML-based detection (Isolation Forest, Local Outlier Factor)
- Severity classification
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from core.anomaly_detection import (
    AnomalyDetector,
    AnomalyResult,
    AnomalyStatistics,
    StatisticalAnomalyDetector,
    TimeSeriesAnomalyDetector,
    MLAnomalyDetector,
    calculate_anomaly_statistics,
)


class TestStatisticalAnomalyDetector:
    """Test statistical anomaly detection methods."""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normally distributed data."""
        np.random.seed(42)
        return np.random.normal(100, 10, 100)
    
    @pytest.fixture
    def data_with_outliers(self):
        """Generate data with outliers."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        # Add some outliers
        data[10] = 200
        data[50] = -50
        data[75] = 150
        return data
    
    def test_zscore_method(self, normal_data):
        """Test z-score anomaly detection."""
        anomaly_mask, z_scores = StatisticalAnomalyDetector.zscore_method(
            normal_data, threshold=3.0
        )
        
        assert len(anomaly_mask) == len(normal_data)
        assert len(z_scores) == len(normal_data)
        
        # For normal data, should detect very few anomalies
        assert anomaly_mask.sum() < 5
        
        # Z-scores should be reasonable
        assert np.all(np.abs(z_scores) < 10)
    
    def test_zscore_detects_outliers(self, data_with_outliers):
        """Test that z-score detects injected outliers."""
        anomaly_mask, z_scores = StatisticalAnomalyDetector.zscore_method(
            data_with_outliers, threshold=2.5
        )
        
        # Should detect at least some outliers
        assert anomaly_mask[10] or anomaly_mask[50] or anomaly_mask[75]
    
    def test_modified_zscore(self, data_with_outliers):
        """Test modified z-score method."""
        anomaly_mask, z_scores = StatisticalAnomalyDetector.modified_zscore_method(
            data_with_outliers, threshold=3.5
        )
        
        assert len(anomaly_mask) == len(data_with_outliers)
        # Modified z-score should be robust to outliers
        assert np.isfinite(z_scores).all()
    
    def test_iqr_method(self, data_with_outliers):
        """Test IQR method."""
        anomaly_mask, outlier_scores = StatisticalAnomalyDetector.iqr_method(
            data_with_outliers, k=1.5
        )
        
        assert len(anomaly_mask) == len(data_with_outliers)
        # Should detect the injected outliers
        assert anomaly_mask[10] or anomaly_mask[50] or anomaly_mask[75]
        # Outlier scores should be non-negative
        assert np.all(outlier_scores >= 0)
    
    def test_iqr_zero_std(self):
        """Test IQR with constant data."""
        constant_data = np.array([100] * 20)
        anomaly_mask, scores = StatisticalAnomalyDetector.iqr_method(constant_data)
        
        # Should handle gracefully
        assert len(anomaly_mask) == len(constant_data)


class TestTimeSeriesAnomalyDetector:
    """Test time series anomaly detection."""
    
    @pytest.fixture
    def smooth_series(self):
        """Generate smooth time series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        values = 100 + 5 * np.sin(np.arange(100) / 10)
        values += np.random.normal(0, 0.5, len(values))
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def series_with_level_shift(self):
        """Generate series with level shift."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        values = np.concatenate([
            np.repeat(100.0, 50),
            np.repeat(120.0, 50),
        ])
        values = values + np.random.normal(0, 0.5, len(values))
        return pd.Series(values, index=dates)
    
    def test_detect_level_shift(self, series_with_level_shift):
        """Test level shift detection."""
        shifts = TimeSeriesAnomalyDetector.detect_level_shift(
            series_with_level_shift, window=10, threshold=2.0
        )
        
        assert len(shifts) == len(series_with_level_shift)
        # Should detect shift around index 50
        assert shifts[45:55].any()
    
    def test_detect_trend_change(self, smooth_series):
        """Test trend change detection."""
        trends = TimeSeriesAnomalyDetector.detect_trend_change(
            smooth_series, window=10, threshold=2.0
        )
        
        assert len(trends) == len(smooth_series)
        # Return type should be boolean array
        assert isinstance(trends, np.ndarray)
    
    def test_calculate_residuals(self, smooth_series):
        """Test residual calculation."""
        residuals = TimeSeriesAnomalyDetector.calculate_residuals(
            smooth_series, window=10
        )
        
        assert len(residuals) == len(smooth_series)
        # Residuals should have zero mean (approximately)
        assert abs(np.mean(residuals)) < 1.0


class TestMLAnomalyDetector:
    """Test machine learning anomaly detection."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for training."""
        np.random.seed(42)
        # Normal data: 100 samples with 2 features
        # First feature centered at 100 with std 5
        # Second feature centered at 10 with std 5
        feature1 = np.random.normal(100, 5, 100)
        feature2 = np.random.normal(10, 5, 100)
        data = np.column_stack([feature1, feature2])
        return data
    
    @pytest.fixture
    def sample_data_with_anomalies(self, sample_data):
        """Generate data with anomalies."""
        data = sample_data.copy()
        # Add anomalies
        data[10] = [200, 50]
        data[50] = [-50, -30]
        return data
    
    def test_isolation_forest_initialization(self):
        """Test Isolation Forest initialization."""
        detector = MLAnomalyDetector(method="isolation_forest", contamination=0.05)
        
        assert detector.method == "isolation_forest"
        assert not detector.is_fit
    
    def test_lof_initialization(self):
        """Test LOF initialization."""
        detector = MLAnomalyDetector(method="lof", contamination=0.05)
        
        assert detector.method == "lof"
        assert not detector.is_fit
    
    def test_fit_isolation_forest(self, sample_data):
        """Test fitting Isolation Forest."""
        detector = MLAnomalyDetector(method="isolation_forest")
        detector.fit(sample_data)
        
        assert detector.is_fit
        assert detector.model is not None
    
    def test_fit_lof(self, sample_data):
        """Test fitting LOF."""
        detector = MLAnomalyDetector(method="lof")
        detector.fit(sample_data)
        
        assert detector.is_fit
        assert detector.model is not None
    
    def test_predict_isolation_forest(self, sample_data_with_anomalies):
        """Test prediction with Isolation Forest."""
        # Split into train/test
        train_data = sample_data_with_anomalies[:80]
        test_data = sample_data_with_anomalies[80:]
        
        detector = MLAnomalyDetector(method="isolation_forest", contamination=0.1)
        detector.fit(train_data)
        
        predictions, scores = detector.predict(test_data)
        
        assert len(predictions) == len(test_data)
        assert len(scores) == len(test_data)
        # Scores should be between 0 and 1
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_predict_without_fit(self, sample_data):
        """Test prediction without fitting raises error."""
        detector = MLAnomalyDetector()
        
        with pytest.raises(ValueError, match="not fit"):
            detector.predict(sample_data)
    
    def test_invalid_method(self):
        """Test initialization with invalid method."""
        detector = MLAnomalyDetector(method="invalid_method")
        
        with pytest.raises(ValueError, match="Unknown method"):
            detector.fit(np.random.randn(50, 2))


class TestAnomalyDetector:
    """Test the main anomaly detector."""
    
    @pytest.fixture
    def price_series(self):
        """Generate price series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.normal(0.1, 2, 100))
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def price_series_with_anomalies(self, price_series):
        """Generate price series with anomalies."""
        series = price_series.copy()
        # Inject anomalies
        series.iloc[20] = series.iloc[20] * 2  # Price spike
        series.iloc[50] = series.iloc[50] * 0.5  # Price drop
        return series
    
    def test_zscore_detector_initialization(self):
        """Test z-score anomaly detector initialization."""
        detector = AnomalyDetector(method="zscore", zscore_threshold=3.0)
        
        assert detector.method == "zscore"
        assert not detector.is_fit
    
    def test_iqr_detector_initialization(self):
        """Test IQR anomaly detector initialization."""
        detector = AnomalyDetector(method="iqr")
        
        assert detector.method == "iqr"
    
    def test_level_shift_detector_initialization(self):
        """Test level shift detector initialization."""
        detector = AnomalyDetector(method="level_shift")
        
        assert detector.method == "level_shift"
    
    def test_fit_ml_detector(self, price_series):
        """Test fitting ML-based detector."""
        detector = AnomalyDetector(method="ml_isolation_forest")
        detector.fit(price_series)
        
        assert detector.is_fit
    
    def test_detect_zscore(self, price_series_with_anomalies):
        """Test anomaly detection with z-score."""
        detector = AnomalyDetector(method="zscore", zscore_threshold=2.0)
        anomalies = detector.detect(price_series_with_anomalies)
        
        # Should detect some anomalies
        assert len(anomalies) > 0
        
        # Check AnomapyResult structure
        if anomalies:
            a = anomalies[0]
            assert isinstance(a, AnomalyResult)
            assert hasattr(a, "timestamp")
            assert hasattr(a, "value")
            assert hasattr(a, "is_anomaly")
            assert hasattr(a, "severity")
    
    def test_detect_iqr(self, price_series_with_anomalies):
        """Test detection with IQR method."""
        detector = AnomalyDetector(method="iqr")
        anomalies = detector.detect(price_series_with_anomalies)
        
        assert len(anomalies) >= 0  # May or may not detect
        
        # All results should be valid
        for anomaly in anomalies:
            assert anomaly.is_anomaly
            assert anomaly.severity in ["low", "medium", "high", "critical"]
    
    def test_detect_level_shift(self, price_series_with_anomalies):
        """Test level shift detection."""
        detector = AnomalyDetector(method="level_shift")
        anomalies = detector.detect(price_series_with_anomalies)
        
        assert isinstance(anomalies, list)
        assert all(isinstance(a, AnomalyResult) for a in anomalies)
    
    def test_detect_ml(self, price_series_with_anomalies):
        """Test ML-based detection."""
        detector = AnomalyDetector(method="ml_isolation_forest", contamination=0.1)
        detector.fit(price_series_with_anomalies[:80])
        
        anomalies = detector.detect(price_series_with_anomalies)
        
        # Should detect some anomalies
        assert len(anomalies) > 0
    
    def test_severity_classification(self, price_series_with_anomalies):
        """Test severity classification."""
        detector = AnomalyDetector(method="zscore", zscore_threshold=2.0)
        anomalies = detector.detect(price_series_with_anomalies)
        
        if anomalies:
            severities = [a.severity for a in anomalies]
            assert all(s in ["low", "medium", "high", "critical"] for s in severities)
    
    def test_anomaly_result_to_dict(self, price_series_with_anomalies):
        """Test AnomalyResult serialization."""
        detector = AnomalyDetector(method="zscore")
        anomalies = detector.detect(price_series_with_anomalies)
        
        if anomalies:
            result_dict = anomalies[0].to_dict()
            assert isinstance(result_dict, dict)
            assert "value" in result_dict
            assert "anomaly_score" in result_dict
            assert "severity" in result_dict


class TestAnomalyStatistics:
    """Test anomaly statistics calculation."""
    
    @pytest.fixture
    def detector_and_series(self):
        """Create detector and series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 + np.random.normal(0, 5, 100)
        series = pd.Series(prices, index=dates)
        
        detector = AnomalyDetector(method="zscore", zscore_threshold=3.0)
        
        return detector, series
    
    def test_calculate_statistics(self, detector_and_series):
        """Test anomaly statistics calculation."""
        detector, series = detector_and_series
        
        stats = calculate_anomaly_statistics(series, detector)
        
        assert isinstance(stats, AnomalyStatistics)
        assert stats.total_points == len(series)
        assert stats.anomaly_rate >= 0
        assert stats.anomaly_rate <= 1
        assert stats.high_severity_count >= 0
        assert stats.mean_anomaly_score >= 0
    
    def test_statistics_to_dict(self, detector_and_series):
        """Test AnomalyStatistics serialization."""
        detector, series = detector_and_series
        
        stats = calculate_anomaly_statistics(series, detector)
        stats_dict = stats.to_dict()
        
        assert isinstance(stats_dict, dict)
        assert "anomaly_rate" in stats_dict
        assert "anomaly_count" in stats_dict


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def realistic_price_series(self):
        """Generate realistic price series."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        np.random.seed(123)
        prices = 100 * (1 + np.random.normal(0.0005, 0.015, 252)).cumprod()
        return pd.Series(prices, index=dates)
    
    def test_compare_detection_methods(self, realistic_price_series):
        """Test comparing different detection methods."""
        methods = ["zscore", "iqr", "level_shift"]
        
        for method in methods:
            detector = AnomalyDetector(method=method)
            
            if method == "level_shift":
                # level_shift doesn't need fitting
                pass
            else:
                detector.fit(realistic_price_series)
            
            anomalies = detector.detect(realistic_price_series)
            
            # Should return list of results
            assert isinstance(anomalies, list)
            assert all(isinstance(a, AnomalyResult) for a in anomalies)
    
    def test_ml_detection_pipeline(self, realistic_price_series):
        """Test ML detection with more realistic data."""
        # Corrupting 5% of data with anomalies
        corrupt_series = realistic_price_series.copy()
        anomaly_indices = np.random.choice(len(corrupt_series), size=int(0.05 * len(corrupt_series)), replace=False)
        
        for idx in anomaly_indices[:10]:
            corrupt_series.iloc[idx] *= np.random.choice([0.5, 2.0])
        
        # Train on first 200 points, test on last 52
        train_series = corrupt_series.iloc[:200]
        test_series = corrupt_series.iloc[200:]
        
        detector = AnomalyDetector(method="ml_isolation_forest", contamination=0.1)
        detector.fit(train_series)
        
        anomalies = detector.detect(test_series)
        
        assert isinstance(anomalies, list)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([], dtype=float)
        detector = AnomalyDetector(method="zscore")
        
        anomalies = detector.detect(series)
        assert len(anomalies) == 0
    
    def test_single_point(self):
        """Test with single data point."""
        dates = pd.date_range(start="2024-01-01", periods=1)
        series = pd.Series([100], index=dates)
        
        detector = AnomalyDetector(method="zscore")
        anomalies = detector.detect(series)
        
        # Should handle gracefully
        assert len(anomalies) >= 0
    
    def test_constant_values(self):
        """Test with constant values."""
        series = pd.Series([100] * 50)
        detector = AnomalyDetector(method="zscore")
        
        anomalies = detector.detect(series)
        
        # Constant series has no anomalies
        assert len(anomalies) == 0
    
    def test_extreme_values(self):
        """Test with extreme values."""
        values = [1e-10, 1e10, 100, 0, -1e10]
        dates = pd.date_range(start="2024-01-01", periods=len(values))
        series = pd.Series(values, index=dates)
        
        detector = AnomalyDetector(method="iqr")
        
        # Should handle without error
        anomalies = detector.detect(series)
        assert isinstance(anomalies, list)
