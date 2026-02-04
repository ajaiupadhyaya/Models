"""
Model Performance Monitoring and Auto-Retraining
Tracks model performance and automatically retrains when needed
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """
    Monitor model performance and trigger retraining.
    Tracks accuracy, returns, Sharpe ratio, and other metrics.
    """
    
    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize monitor.
        
        Args:
            models_dir: Directory to store model performance data
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_history = {}
        self.retrain_thresholds = {
            'accuracy_drop': 0.05,  # 5% accuracy drop
            'sharpe_drop': 0.2,  # 20% Sharpe drop
            'max_drawdown': 0.15,  # 15% max drawdown
            'days_since_retrain': 7  # Retrain after 7 days
        }
    
    def record_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: float,
        actual: float,
        timestamp: datetime = None
    ):
        """
        Record a prediction and actual outcome.
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
            prediction: Predicted value
            actual: Actual value
            timestamp: Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        key = f"{model_name}_{symbol}"
        
        if key not in self.performance_history:
            self.performance_history[key] = {
                'predictions': [],
                'actuals': [],
                'timestamps': [],
                'metrics': {}
            }
        
        self.performance_history[key]['predictions'].append(prediction)
        self.performance_history[key]['actuals'].append(actual)
        self.performance_history[key]['timestamps'].append(timestamp)
        
        # Keep only last 1000 predictions
        if len(self.performance_history[key]['predictions']) > 1000:
            self.performance_history[key]['predictions'] = \
                self.performance_history[key]['predictions'][-1000:]
            self.performance_history[key]['actuals'] = \
                self.performance_history[key]['actuals'][-1000:]
            self.performance_history[key]['timestamps'] = \
                self.performance_history[key]['timestamps'][-1000:]
        
        # Update metrics
        self._update_metrics(key)
    
    def _update_metrics(self, key: str):
        """Update performance metrics for a model."""
        data = self.performance_history[key]
        
        if len(data['predictions']) < 10:
            return
        
        predictions = np.array(data['predictions'])
        actuals = np.array(data['actuals'])
        
        # Calculate accuracy (for direction prediction)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        accuracy = np.mean(pred_direction == actual_direction)
        
        # Calculate returns and metrics via canonical core.utils
        from core.utils import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
        )
        returns_series = pd.Series(actuals)
        cumulative_returns = (1 + returns_series).cumprod()
        total_return = float(cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0.0
        sharpe = float(calculate_sharpe_ratio(returns_series)) if len(returns_series) > 1 else 0.0
        if not np.isfinite(sharpe):
            sharpe = 0.0
        max_drawdown = float(calculate_max_drawdown(returns_series)) if len(returns_series) > 0 else 0.0
        
        # Calculate MAE and RMSE
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        data['metrics'] = {
            'accuracy': float(accuracy),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'mae': float(mae),
            'rmse': float(rmse),
            'total_predictions': len(predictions),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_performance(self, model_name: str, symbol: str) -> Optional[Dict]:
        """
        Get performance metrics for a model.
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
        
        Returns:
            Performance metrics dictionary
        """
        key = f"{model_name}_{symbol}"
        if key in self.performance_history:
            return self.performance_history[key]['metrics']
        return None
    
    def should_retrain(self, model_name: str, symbol: str) -> Tuple[bool, str]:
        """
        Check if model should be retrained.
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
        
        Returns:
            (should_retrain, reason)
        """
        key = f"{model_name}_{symbol}"
        
        if key not in self.performance_history:
            return False, "No performance data"
        
        metrics = self.performance_history[key]['metrics']
        
        if not metrics:
            return False, "No metrics available"
        
        # Check accuracy drop
        if 'baseline_accuracy' in metrics:
            accuracy_drop = metrics['baseline_accuracy'] - metrics['accuracy']
            if accuracy_drop > self.retrain_thresholds['accuracy_drop']:
                return True, f"Accuracy dropped by {accuracy_drop:.2%}"
        
        # Check Sharpe ratio drop
        if 'baseline_sharpe' in metrics:
            sharpe_drop = metrics['baseline_sharpe'] - metrics['sharpe_ratio']
            if sharpe_drop > self.retrain_thresholds['sharpe_drop']:
                return True, f"Sharpe ratio dropped by {sharpe_drop:.2f}"
        
        # Check max drawdown
        if abs(metrics['max_drawdown']) > self.retrain_thresholds['max_drawdown']:
            return True, f"Max drawdown exceeded: {metrics['max_drawdown']:.2%}"
        
        # Check days since last retrain
        if 'last_retrain' in metrics:
            last_retrain = datetime.fromisoformat(metrics['last_retrain'])
            days_since = (datetime.now() - last_retrain).days
            if days_since >= self.retrain_thresholds['days_since_retrain']:
                return True, f"Days since retrain: {days_since}"
        
        return False, "Performance acceptable"
    
    def set_baseline(self, model_name: str, symbol: str):
        """
        Set baseline metrics for comparison.
        Should be called after initial training.
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
        """
        key = f"{model_name}_{symbol}"
        if key in self.performance_history:
            metrics = self.performance_history[key]['metrics']
            metrics['baseline_accuracy'] = metrics.get('accuracy', 0)
            metrics['baseline_sharpe'] = metrics.get('sharpe_ratio', 0)
            metrics['last_retrain'] = datetime.now().isoformat()
    
    def save_performance(self):
        """Save performance history to disk."""
        try:
            performance_file = self.models_dir / "performance_history.json"
            # Convert to JSON-serializable format
            serializable = {}
            for key, value in self.performance_history.items():
                serializable[key] = {
                    'metrics': value['metrics'],
                    'total_predictions': len(value['predictions'])
                }
            
            with open(performance_file, 'w') as f:
                json.dump(serializable, f, indent=2)
            
            logger.info("Performance history saved")
        except Exception as e:
            logger.error(f"Failed to save performance: {e}")
    
    def load_performance(self):
        """Load performance history from disk."""
        try:
            performance_file = self.models_dir / "performance_history.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                    # Restore structure (without full prediction history)
                    for key, value in data.items():
                        if key not in self.performance_history:
                            self.performance_history[key] = {
                                'predictions': [],
                                'actuals': [],
                                'timestamps': [],
                                'metrics': value.get('metrics', {})
                            }
                
                logger.info("Performance history loaded")
        except Exception as e:
            logger.error(f"Failed to load performance: {e}")
