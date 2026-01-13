"""
ML Models for Prediction and Trading
LSTM, Ensemble, and RL-ready models for market prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


class LSTMPredictor:
    """
    LSTM-based time series prediction model.
    Predicts next-period returns using LSTM neural network.
    """
    
    def __init__(self, lookback_window: int = 20, hidden_units: int = 64):
        """
        Initialize LSTM predictor.
        
        Args:
            lookback_window: Number of past periods to use
            hidden_units: LSTM hidden units
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for LSTM. Install: pip install tensorflow")
        
        self.lookback_window = lookback_window
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            X, y arrays
        """
        # Extract price and volume features
        prices = df['Close'].values.reshape(-1, 1)
        volumes = df['Volume'].values.reshape(-1, 1)
        high_low = ((df['High'] - df['Low']) / df['Close']).values.reshape(-1, 1)
        
        # Combine features
        features = np.hstack([prices, volumes, high_low])
        features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.lookback_window):
            X.append(features[i:i + self.lookback_window])
            y.append(df['Close'].pct_change().iloc[i + self.lookback_window])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model architecture."""
        self.model = Sequential([
            LSTM(self.hidden_units, input_shape=(self.lookback_window, 3), return_sequences=True),
            Dropout(0.2),
            LSTM(self.hidden_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')  # Output between -1 and 1
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """
        Train the LSTM model.
        
        Args:
            df: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        X, y = self.prepare_data(df)
        
        if self.model is None:
            self.build_model()
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.2
        )
        
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            df: Data to predict on
        
        Returns:
            Array of signals (-1 to 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        prices = df['Close'].values.reshape(-1, 1)
        volumes = df['Volume'].values.reshape(-1, 1)
        high_low = ((df['High'] - df['Low']) / df['Close']).values.reshape(-1, 1)
        
        features = np.hstack([prices, volumes, high_low])
        features = self.scaler.transform(features)
        
        # Create sequences for prediction
        predictions = np.zeros(len(df))
        
        for i in range(self.lookback_window, len(df)):
            X_seq = features[i - self.lookback_window:i].reshape(1, self.lookback_window, 3)
            pred = self.model.predict(X_seq, verbose=0)[0, 0]
            predictions[i] = np.clip(pred, -1, 1)
        
        return predictions


class EnsemblePredictor:
    """
    Ensemble of multiple ML models for robust predictions.
    Combines Random Forest, Gradient Boosting, and simple rules.
    """
    
    def __init__(self, lookback_window: int = 20):
        """
        Initialize ensemble predictor.
        
        Args:
            lookback_window: Number of periods for feature calculation
        """
        self.lookback_window = lookback_window
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate feature matrix from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Feature matrix
        """
        features_list = []
        
        # Price momentum
        returns = df['Close'].pct_change()
        features_list.append(returns.values)
        
        # Moving averages
        sma_5 = df['Close'].rolling(5).mean()
        sma_20 = df['Close'].rolling(20).mean()
        features_list.append((df['Close'] / sma_5 - 1).values)
        features_list.append((df['Close'] / sma_20 - 1).values)
        
        # Volatility
        vol = df['Close'].pct_change().rolling(20).std()
        features_list.append(vol.values)
        
        # Volume
        volume_sma = df['Volume'].rolling(20).mean()
        features_list.append((df['Volume'] / volume_sma).values)
        
        # High-Low range
        features_list.append(((df['High'] - df['Low']) / df['Close']).values)
        
        # RSI components
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features_list.append((rsi / 100).values)
        
        # Stack features
        features = np.column_stack(features_list)
        features = np.nan_to_num(features)
        
        return features
    
    def train(self, df: pd.DataFrame):
        """
        Train ensemble models.
        
        Args:
            df: Training data
        """
        X = self.calculate_features(df)
        y = df['Close'].pct_change().values
        
        # Remove first NaN
        X = X[1:]
        y = y[1:]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train models
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)
        
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            df: Data to predict on
        
        Returns:
            Array of signals (-1 to 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.calculate_features(df)
        X = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)
        
        # Ensemble average (60% GB, 40% RF - GB typically better for this)
        ensemble_pred = 0.6 * gb_pred + 0.4 * rf_pred
        
        # Clip to [-1, 1]
        signals = np.clip(ensemble_pred, -1, 1)
        
        return signals


class RLReadyEnvironment:
    """
    OpenAI Gym-compatible environment for RL trading.
    Can be used with stable-baselines3 and other RL frameworks.
    """
    
    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize RL environment.
        
        Args:
            df: Price data
            initial_capital: Starting capital
        """
        self.df = df
        self.initial_capital = initial_capital
        self.current_step = 0
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.capital = initial_capital
        self.entry_price = None
        self.history = []
        
        # State space
        self.lookback = 20
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = self.lookback
        self.position = 0
        self.capital = self.initial_capital
        self.entry_price = None
        self.history = []
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state."""
        start_idx = max(0, self.current_step - self.lookback)
        price_window = self.df['Close'].iloc[start_idx:self.current_step].values
        
        # Normalize prices
        price_norm = (price_window - price_window.mean()) / (price_window.std() + 1e-6)
        
        # Add other features
        current_price = self.df['Close'].iloc[self.current_step]
        volume_ratio = self.df['Volume'].iloc[self.current_step] / self.df['Volume'].rolling(20).mean().iloc[self.current_step]
        
        state = np.concatenate([
            price_norm,
            [volume_ratio],
            [self.position],
            [self.capital / self.initial_capital]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: 0=hold, 1=go long, 2=go short, 3=close position
        
        Returns:
            state, reward, done, info
        """
        current_price = self.df['Close'].iloc[self.current_step]
        
        reward = 0
        
        # Process action
        if action == 1:  # Go long
            if self.position != 1:
                if self.position == -1:  # Close short first
                    loss = (self.entry_price - current_price) * (self.capital / self.entry_price)
                    self.capital += loss
                    reward -= abs(loss) / self.initial_capital
                
                self.position = 1
                self.entry_price = current_price
        
        elif action == 2:  # Go short
            if self.position != -1:
                if self.position == 1:  # Close long first
                    gain = (current_price - self.entry_price) * (self.capital / self.entry_price)
                    self.capital += gain
                    reward += gain / self.initial_capital
                
                self.position = -1
                self.entry_price = current_price
        
        elif action == 3:  # Close position
            if self.position == 1:
                gain = (current_price - self.entry_price) * (self.capital / self.entry_price)
                self.capital += gain
                reward += gain / self.initial_capital
                self.position = 0
            elif self.position == -1:
                loss = (self.entry_price - current_price) * (self.capital / self.entry_price)
                self.capital += loss
                reward -= abs(loss) / self.initial_capital
                self.position = 0
        
        # Mark-to-market reward
        if self.position == 1:
            mtm_value = self.capital + (current_price - self.entry_price) * (self.capital / self.entry_price)
            reward += (mtm_value - self.capital) / self.initial_capital * 0.1
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        self.history.append({
            'step': self.current_step,
            'action': action,
            'price': current_price,
            'position': self.position,
            'capital': self.capital,
            'reward': reward
        })
        
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        info = {'capital': self.capital, 'position': self.position}
        
        return next_state, reward, done, info
    
    def get_performance(self) -> Dict:
        """Get environment performance metrics."""
        return {
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'trades': len([h for h in self.history if h['action'] in [1, 2]]),
            'history': self.history
        }
