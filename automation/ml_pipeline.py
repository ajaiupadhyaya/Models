"""
ML/AI/RL/DL Pipeline
Comprehensive machine learning automation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging
import pickle
import sys
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_fetcher import DataFetcher
from models.ml.forecasting import TimeSeriesForecaster
from models.ml.advanced_trading import LSTMPredictor, EnsemblePredictor

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Comprehensive ML/AI/RL/DL pipeline for automated model training and prediction.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize ML pipeline.
        
        Args:
            models_dir: Directory for saving models
        """
        self.models_dir = models_dir or project_root / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_fetcher = DataFetcher()
        self.trained_models: Dict[str, Any] = {}
        
        logger.info(f"ML Pipeline initialized: {self.models_dir}")
    
    def train_forecasting_model(self,
                                symbol: str,
                                model_type: str = 'random_forest',
                                lookback: int = 20,
                                forecast_periods: int = 30) -> Dict[str, Any]:
        """
        Train time series forecasting model.
        
        Args:
            symbol: Stock symbol
            model_type: Type of model ('random_forest', 'gradient_boosting', 'neural_network')
            lookback: Number of lag features
            forecast_periods: Number of periods to forecast
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training {model_type} forecasting model for {symbol}...")
            
            # Fetch data
            data = self.data_fetcher.get_stock_data(symbol, period='2y')
            
            if len(data) < 100:
                return {
                    'success': False,
                    'error': f'Insufficient data: {len(data)} rows'
                }
            
            # Prepare data
            prices = data['Close']
            
            # Train model
            forecaster = TimeSeriesForecaster(model_type=model_type)
            forecaster.fit(prices, n_lags=lookback)
            
            # Generate forecast
            forecast = forecaster.predict(prices, n_periods=forecast_periods)
            
            # Evaluate
            train_pred = forecaster.predict(prices.iloc[:-forecast_periods], n_periods=forecast_periods)
            actual = prices.iloc[-forecast_periods:].values
            
            mae = np.mean(np.abs(train_pred - actual))
            mse = np.mean((train_pred - actual) ** 2)
            rmse = np.sqrt(mse)
            
            # Save model
            model_file = self.models_dir / f"{symbol}_forecast_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(forecaster, f)
            
            model_id = f"{symbol}_forecast_{model_type}"
            self.trained_models[model_id] = {
                'model': forecaster,
                'model_file': str(model_file),
                'trained_date': datetime.now(),
                'metrics': {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse)
                }
            }
            
            return {
                'success': True,
                'symbol': symbol,
                'model_type': model_type,
                'model_id': model_id,
                'forecast': forecast.tolist(),
                'metrics': {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse)
                },
                'model_file': str(model_file)
            }
            
        except Exception as e:
            logger.error(f"Error training forecasting model for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_lstm_model(self,
                         symbol: str,
                         lookback_window: int = 20,
                         hidden_units: int = 64,
                         epochs: int = 50) -> Dict[str, Any]:
        """
        Train LSTM deep learning model.
        
        Args:
            symbol: Stock symbol
            lookback_window: LSTM lookback window
            hidden_units: Number of LSTM hidden units
            epochs: Training epochs
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training LSTM model for {symbol}...")
            
            # Fetch data
            data = self.data_fetcher.get_stock_data(symbol, period='2y')
            
            if len(data) < 200:
                return {
                    'success': False,
                    'error': f'Insufficient data: {len(data)} rows (need 200+)'
                }
            
            # Train LSTM
            lstm = LSTMPredictor(
                lookback_window=lookback_window,
                hidden_units=hidden_units
            )
            lstm.train(data, epochs=epochs, batch_size=32)
            
            # Test prediction
            test_pred = lstm.predict(data.tail(50))
            
            # Save model
            model_file = self.models_dir / f"{symbol}_lstm_{datetime.now().strftime('%Y%m%d')}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(lstm, f)
            
            model_id = f"{symbol}_lstm"
            self.trained_models[model_id] = {
                'model': lstm,
                'model_file': str(model_file),
                'trained_date': datetime.now(),
                'is_trained': True
            }
            
            return {
                'success': True,
                'symbol': symbol,
                'model_type': 'lstm',
                'model_id': model_id,
                'test_prediction': test_pred.tolist() if hasattr(test_pred, 'tolist') else test_pred,
                'model_file': str(model_file)
            }
            
        except ImportError as e:
            logger.error(f"TensorFlow not available: {e}")
            return {
                'success': False,
                'error': 'TensorFlow required for LSTM. Install: pip install tensorflow'
            }
        except Exception as e:
            logger.error(f"Error training LSTM for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_ensemble_model(self, symbol: str) -> Dict[str, Any]:
        """
        Train ensemble ML model.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training ensemble model for {symbol}...")
            
            # Fetch data
            data = self.data_fetcher.get_stock_data(symbol, period='2y')
            
            if len(data) < 100:
                return {
                    'success': False,
                    'error': f'Insufficient data: {len(data)} rows'
                }
            
            # Train ensemble
            ensemble = EnsemblePredictor()
            ensemble.train(data)
            
            # Test prediction
            test_pred = ensemble.predict(data.tail(50))
            
            # Save model
            model_file = self.models_dir / f"{symbol}_ensemble_{datetime.now().strftime('%Y%m%d')}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(ensemble, f)
            
            model_id = f"{symbol}_ensemble"
            self.trained_models[model_id] = {
                'model': ensemble,
                'model_file': str(model_file),
                'trained_date': datetime.now(),
                'is_trained': True
            }
            
            return {
                'success': True,
                'symbol': symbol,
                'model_type': 'ensemble',
                'model_id': model_id,
                'test_prediction': test_pred.tolist() if hasattr(test_pred, 'tolist') else test_pred,
                'model_file': str(model_file)
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_all_models(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Train all model types for given symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Training results for all models
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = {
                'forecasting': self.train_forecasting_model(symbol),
                'lstm': self.train_lstm_model(symbol, epochs=30),  # Reduced for speed
                'ensemble': self.train_ensemble_model(symbol)
            }
        
        # Summary
        total_models = len(symbols) * 3
        successful = sum(
            1 for sym_results in results.values()
            for model_result in sym_results.values()
            if model_result.get('success', False)
        )
        
        return {
            'summary': {
                'total_models': total_models,
                'successful': successful,
                'failed': total_models - successful
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """
        Make prediction with trained model.
        
        Args:
            model_id: Model identifier
            data: Input data
        
        Returns:
            Predictions
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.trained_models[model_id]['model']
        
        if hasattr(model, 'predict'):
            return model.predict(data)
        else:
            raise ValueError(f"Model {model_id} does not support prediction")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about all trained models.
        
        Returns:
            Model information dictionary
        """
        return {
            'total_models': len(self.trained_models),
            'models': {
                model_id: {
                    'trained_date': str(info['trained_date']),
                    'model_file': info['model_file'],
                    'is_trained': info.get('is_trained', False),
                    'metrics': info.get('metrics', {})
                }
                for model_id, info in self.trained_models.items()
            }
        }


if __name__ == "__main__":
    # Example usage
    pipeline = MLPipeline()
    
    # Train models for a symbol
    result = pipeline.train_all_models(["AAPL"])
    print(json.dumps(result, indent=2, default=str))
