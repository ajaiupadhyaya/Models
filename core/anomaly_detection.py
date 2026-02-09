"""
Anomaly Detection Engine

Machine learning-based anomaly detection for financial data:
- Statistical methods (z-score, IQR, modified z-score)
- Time series decomposition and residual analysis
- Isolation Forest for multivariate anomalies
- Local Outlier Factor (LOF) for density-based detection
- Seasonal adjustment and trend analysis
- Alert threshold management

Usage:
    from core.anomaly_detection import AnomalyDetector, AnomalyResult
    
    detector = AnomalyDetector(method="isolation_forest")
    detector.fit(historical_data)
    anomalies = detector.detect(new_data)
    for anomaly in anomalies:
        print(f"Anomaly at {anomaly.timestamp}: {anomaly.value}")
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Detected anomaly."""
    
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float  # std deviations from expected
    anomaly_score: float  # 0-1, higher = more anomalous
    method: str
    is_anomaly: bool
    severity: str  # "low", "medium", "high", "critical"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "anomaly_score": self.anomaly_score,
            "method": self.method,
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
        }


@dataclass
class AnomalyStatistics:
    """Statistics about detected anomalies."""
    
    total_points: int
    anomaly_count: int
    anomaly_rate: float
    high_severity_count: int
    mean_anomaly_score: float
    max_deviation: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_points": self.total_points,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": self.anomaly_rate,
            "high_severity_count": self.high_severity_count,
            "mean_anomaly_score": self.mean_anomaly_score,
            "max_deviation": self.max_deviation,
        }


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using z-score and IQR methods.
    """
    
    @staticmethod
    def zscore_method(
        data: np.ndarray,
        threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using z-score.
        
        Args:
            data: Data array
            threshold: Z-score threshold (default: 3 = 99.7% CI)
            
        Returns:
            Tuple of (anomaly_mask, z_scores)
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > threshold
        
        return anomaly_mask, z_scores
    
    @staticmethod
    def modified_zscore_method(
        data: np.ndarray,
        threshold: float = 3.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using modified z-score (more robust to outliers).
        
        Args:
            data: Data array
            threshold: Modified z-score threshold
            
        Returns:
            Tuple of (anomaly_mask, modified_z_scores)
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        modified_z_scores = 0.6745 * (data - median) / mad
        anomaly_mask = np.abs(modified_z_scores) > threshold
        
        return anomaly_mask, modified_z_scores
    
    @staticmethod
    def iqr_method(
        data: np.ndarray,
        k: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Interquartile Range (IQR).
        
        Args:
            data: Data array
            k: IQR multiplier (default: 1.5, classic outlier definition)
            
        Returns:
            Tuple of (anomaly_mask, outlier_scores)
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        
        # Calculate outlier score (distance from bounds)
        outlier_scores = np.zeros(len(data))
        below = data < lower_bound
        above = data > upper_bound
        
        outlier_scores[below] = np.abs(data[below] - lower_bound) / (iqr + 1e-10)
        outlier_scores[above] = np.abs(data[above] - upper_bound) / (iqr + 1e-10)
        
        return anomaly_mask, outlier_scores


class TimeSeriesAnomalyDetector:
    """
    Anomaly detection using time series decomposition and residual analysis.
    """
    
    @staticmethod
    def detect_level_shift(
        series: pd.Series,
        window: int = 20,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Detect sudden level shifts in time series.
        
        Args:
            series: Time series data
            window: Window size for comparison
            threshold: Threshold for detecting shift
            
        Returns:
            Anomaly mask
        """
        if len(series) < window * 2:
            return np.zeros(len(series), dtype=bool)
        
        shifts = np.zeros(len(series), dtype=bool)
        
        for i in range(window, len(series) - window):
            before_mean = series.iloc[i-window:i].mean()
            after_mean = series.iloc[i:i+window].mean()
            
            before_std = series.iloc[i-window:i].std()
            after_std = series.iloc[i:i+window].std()
            
            combined_std = np.sqrt((before_std**2 + after_std**2) / 2)
            
            if combined_std > 0:
                shift_magnitude = abs(after_mean - before_mean) / combined_std
                if shift_magnitude > threshold:
                    shifts[i] = True
        
        return shifts
    
    @staticmethod
    def detect_trend_change(
        series: pd.Series,
        window: int = 20,
        threshold: float = 2.0,
    ) -> np.ndarray:
        """
        Detect changes in trend direction/magnitude.
        
        Args:
            series: Time series data
            window: Window size for trend calculation
            threshold: Threshold for detecting trend change
            
        Returns:
            Anomaly mask
        """
        if len(series) < window * 2:
            return np.zeros(len(series), dtype=bool)
        
        # Calculate slopes
        slopes = np.zeros(len(series))
        for i in range(window, len(series)):
            x = np.arange(window)
            y = series.iloc[i-window:i].values
            slope = np.polyfit(x, y, 1)[0]
            slopes[i] = slope
        
        # Detect significant changes in slope
        slope_changes = np.abs(np.diff(slopes))
        
        # Normalize changes
        mean_change = np.mean(slope_changes[window:])
        std_change = np.std(slope_changes[window:])
        
        if std_change == 0:
            return np.zeros(len(series), dtype=bool)
        
        normalized_changes = np.abs(slope_changes - mean_change) / std_change
        
        anomalies = np.zeros(len(series), dtype=bool)
        anomalies[window:] = normalized_changes[window-1:] > threshold
        
        return anomalies
    
    @staticmethod
    def calculate_residuals(
        series: pd.Series,
        window: int = 20,
    ) -> np.ndarray:
        """
        Calculate residuals from moving average trend.
        
        Args:
            series: Time series data
            window: MA window size
            
        Returns:
            Residuals
        """
        ma = series.rolling(window=window, center=True).mean()
        residuals = series - ma
        return residuals.fillna(0).values


class MLAnomalyDetector:
    """
    Machine learning-based anomaly detection.
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
    ):
        """
        Initialize ML anomaly detector.
        
        Args:
            method: Detection method ("isolation_forest" or "lof")
            contamination: Expected proportion of anomalies (0-0.5)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fit = False
    
    def fit(self, X: np.ndarray):
        """
        Fit the anomaly detection model.
        
        Args:
            X: Training data (n_samples, n_features)
        """
        if len(X) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
        elif self.method == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model.fit(X_scaled)
        self.is_fit = True
        
        logger.info(f"Fit {self.method} on {len(X)} samples")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Args:
            X: Data to predict (n_samples, n_features)
            
        Returns:
            Tuple of (predictions[-1 or 1], anomaly_scores)
        """
        if not self.is_fit:
            raise ValueError("Model not fit. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Get scores
        if hasattr(self.model, "score_samples"):
            scores = self.model.score_samples(X_scaled)
            # Normalize scores to 0-1
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            scores = np.abs(predictions)
        
        return predictions, scores


class AnomalyDetector:
    """
    Comprehensive anomaly detector combining multiple methods.
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        zscore_threshold: float = 3.0,
        contamination: float = 0.05,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method
            zscore_threshold: Z-score threshold for statistical method
            contamination: Expected anomaly rate for ML methods
        """
        self.method = method
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.ml_detector = None if method.startswith("stat") else None
        self.is_fit = False
    
    def fit(self, data: pd.Series):
        """
        Fit the detector to historical data.
        
        Args:
            data: Historical price/values
        """
        if self.method.startswith("ml"):
            # Prepare features for ML
            features = self._extract_features(data)
            self.ml_detector = MLAnomalyDetector(method=self.method.replace("ml_", ""))
            self.ml_detector.fit(features)
        
        self.is_fit = True
        logger.info(f"Fit {self.method} on {len(data)} samples")
    
    def _extract_features(self, series: pd.Series) -> np.ndarray:
        """Extract features from time series."""
        returns = series.pct_change().values[1:]
        
        features = []
        for i in range(len(returns)):
            feature_vector = [
                returns[i],
                series.iloc[i],
                (series.iloc[i] / series.iloc[max(0, i-5)]) - 1 if i > 5 else 0,
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def detect(
        self,
        data: pd.Series,
        severity_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in data.
        
        Args:
            data: Data to check for anomalies
            severity_thresholds: Thresholds for severity classification
            
        Returns:
            List of AnomalyResult objects
        """
        if severity_thresholds is None:
            severity_thresholds = {
                "low": 0.25,
                "medium": 0.50,
                "high": 0.75,
                "critical": 0.90,
            }
        
        anomalies = []
        
        if self.method == "zscore":
            anomaly_mask, z_scores = StatisticalAnomalyDetector.zscore_method(
                data.values, self.zscore_threshold
            )
            for i in range(len(data)):
                if anomaly_mask[i]:
                    anomalies.append(AnomalyResult(
                        timestamp=data.index[i],
                        value=float(data.iloc[i]),
                        expected_value=float(np.mean(data)),
                        deviation=float(z_scores[i]),
                        anomaly_score=min(float(z_scores[i]) / self.zscore_threshold, 1.0),
                        method=self.method,
                        is_anomaly=True,
                        severity=self._classify_severity(
                            z_scores[i], severity_thresholds, self.zscore_threshold
                        ),
                    ))
        
        elif self.method == "iqr":
            anomaly_mask, scores = StatisticalAnomalyDetector.iqr_method(data.values)
            for i in range(len(data)):
                if anomaly_mask[i]:
                    anomaly_score = min(scores[i] / 5.0, 1.0)
                    anomalies.append(AnomalyResult(
                        timestamp=data.index[i],
                        value=float(data.iloc[i]),
                        expected_value=float(np.median(data)),
                        deviation=float(scores[i]),
                        anomaly_score=anomaly_score,
                        method=self.method,
                        is_anomaly=True,
                        severity=self._classify_severity_ml(
                            anomaly_score, severity_thresholds
                        ),
                    ))
        
        elif self.method == "level_shift":
            anomaly_mask = TimeSeriesAnomalyDetector.detect_level_shift(data)
            for i in range(len(data)):
                if anomaly_mask[i]:
                    anomalies.append(AnomalyResult(
                        timestamp=data.index[i],
                        value=float(data.iloc[i]),
                        expected_value=float(data.iloc[max(0, i-5):i].mean()),
                        deviation=float(data.iloc[i] - data.iloc[max(0, i-1)]),
                        anomaly_score=0.75,
                        method=self.method,
                        is_anomaly=True,
                        severity="high",
                    ))
        
        elif self.method.startswith("ml"):
            features = self._extract_features(data)
            predictions, scores = self.ml_detector.predict(features)
            
            for i in range(len(scores)):
                is_anomaly = predictions[i] == -1
                if is_anomaly:
                    anomalies.append(AnomalyResult(
                        timestamp=data.index[i+1],  # Features offset by 1
                        value=float(data.iloc[i+1]),
                        expected_value=float(data.iloc[i]) if i > 0 else float(data.iloc[i+1]),
                        deviation=float((data.iloc[i+1] - data.iloc[i]) / data.iloc[i] * 100),
                        anomaly_score=float(scores[i]),
                        method=self.method,
                        is_anomaly=True,
                        severity=self._classify_severity_ml(scores[i], severity_thresholds),
                    ))
        
        return anomalies
    
    @staticmethod
    def _classify_severity(
        deviation: float,
        thresholds: Dict[str, float],
        scaling_factor: float,
    ) -> str:
        """Classify severity based on deviation."""
        normalized_score = deviation / scaling_factor
        
        if normalized_score > thresholds.get("critical", 0.9):
            return "critical"
        elif normalized_score > thresholds.get("high", 0.75):
            return "high"
        elif normalized_score > thresholds.get("medium", 0.5):
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _classify_severity_ml(
        score: float,
        thresholds: Dict[str, float],
    ) -> str:
        """Classify severity based on ML anomaly score."""
        if score > thresholds.get("critical", 0.9):
            return "critical"
        elif score > thresholds.get("high", 0.75):
            return "high"
        elif score > thresholds.get("medium", 0.5):
            return "medium"
        else:
            return "low"


def calculate_anomaly_statistics(
    data: pd.Series,
    detector: AnomalyDetector,
) -> AnomalyStatistics:
    """
    Calculate anomaly statistics.
    
    Args:
        data: Time series data
        detector: Fitted anomaly detector
        
    Returns:
        AnomalyStatistics object
    """
    anomalies = detector.detect(data)
    
    high_severity = sum(1 for a in anomalies if a.severity in ["high", "critical"])
    
    return AnomalyStatistics(
        total_points=len(data),
        anomaly_count=len(anomalies),
        anomaly_rate=len(anomalies) / len(data) if len(data) > 0 else 0,
        high_severity_count=high_severity,
        mean_anomaly_score=np.mean([a.anomaly_score for a in anomalies]) if anomalies else 0,
        max_deviation=max(a.deviation for a in anomalies) if anomalies else 0,
    )
