"""
Advanced ML Feature Engineering for financial time series.
Phase 2 - Awesome Quant Integration
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class LabelGenerator:
    """
    Generate labels for supervised ML models using advanced techniques.
    Supports triple-barrier method and meta-labeling.
    """
    
    @staticmethod
    def fixed_horizon_labels(
        returns: pd.Series,
        horizon: int = 5,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Generate labels based on fixed-horizon returns.
        
        Args:
            returns: Price returns series
            horizon: Periods to look ahead
            threshold: Return threshold for classification (0 = sign-based)
        
        Returns:
            Series with labels: 1 (up), -1 (down), 0 (neutral)
        """
        # Calculate forward returns
        forward_returns = returns.shift(-horizon)
        
        # Classify
        if threshold == 0:
            # Simple sign-based
            labels = forward_returns.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        else:
            # Threshold-based
            labels = forward_returns.apply(
                lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
            )
        
        return labels
    
    @staticmethod
    def triple_barrier_labels(
        prices: pd.Series,
        target_profit: float = 0.02,
        stop_loss: float = 0.01,
        max_holding_period: int = 20,
        side_predictions: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate labels using triple-barrier method.
        
        The triple-barrier method sets:
        1. Upper barrier: profit target
        2. Lower barrier: stop-loss
        3. Vertical barrier: maximum holding period
        
        Label is determined by which barrier is touched first.
        
        Args:
            prices: Price series
            target_profit: Profit target (e.g., 0.02 = 2%)
            stop_loss: Stop-loss threshold (e.g., 0.01 = 1%)
            max_holding_period: Maximum holding period (bars)
            side_predictions: Optional primary model predictions (1=long, -1=short)
        
        Returns:
            DataFrame with columns: label, returns, holding_period, barrier_touched
        """
        results = []
        
        for i in range(len(prices) - max_holding_period):
            entry_price = prices.iloc[i]
            
            # Determine side (default to long if no predictions)
            side = 1 if side_predictions is None else side_predictions.iloc[i]
            if pd.isna(side):
                side = 1
            
            # Set barriers based on side
            if side == 1:  # Long position
                upper_barrier = entry_price * (1 + target_profit)
                lower_barrier = entry_price * (1 - stop_loss)
            else:  # Short position
                upper_barrier = entry_price * (1 - target_profit)
                lower_barrier = entry_price * (1 + stop_loss)
            
            # Check future prices
            future_prices = prices.iloc[i+1:i+1+max_holding_period]
            
            # Find which barrier is hit first
            label = 0
            returns = 0
            holding_period = max_holding_period
            barrier_touched = 'time'
            
            for j, price in enumerate(future_prices, start=1):
                if side == 1:
                    # Long position
                    if price >= upper_barrier:
                        label = 1
                        returns = (price - entry_price) / entry_price
                        holding_period = j
                        barrier_touched = 'profit'
                        break
                    elif price <= lower_barrier:
                        label = -1
                        returns = (price - entry_price) / entry_price
                        holding_period = j
                        barrier_touched = 'stop'
                        break
                else:
                    # Short position
                    if price <= upper_barrier:
                        label = 1
                        returns = (entry_price - price) / entry_price
                        holding_period = j
                        barrier_touched = 'profit'
                        break
                    elif price >= lower_barrier:
                        label = -1
                        returns = (entry_price - price) / entry_price
                        holding_period = j
                        barrier_touched = 'stop'
                        break
            
            # If no barrier hit, use time barrier
            if label == 0:
                exit_price = future_prices.iloc[-1]
                if side == 1:
                    returns = (exit_price - entry_price) / entry_price
                else:
                    returns = (entry_price - exit_price) / entry_price
                label = 1 if returns > 0 else -1
            
            results.append({
                'label': label,
                'returns': returns,
                'holding_period': holding_period,
                'barrier_touched': barrier_touched
            })
        
        labels_df = pd.DataFrame(results, index=prices.index[:len(results)])
        return labels_df
    
    @staticmethod
    def meta_labeling(
        primary_predictions: pd.Series,
        actual_returns: pd.Series,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Generate meta-labels for position sizing model.
        
        Meta-labeling treats ML as a bet sizing problem:
        - Primary model predicts direction (long/short)
        - Meta-model predicts probability of primary model being correct
        - Meta-labels: 1 = primary model correct, 0 = incorrect
        
        Args:
            primary_predictions: Primary model predictions (1=long, -1=short)
            actual_returns: Actual forward returns
            threshold: Minimum return to consider "correct" (0 = any profit)
        
        Returns:
            Series of meta-labels (1=correct, 0=incorrect)
        """
        # Align indices
        common_idx = primary_predictions.index.intersection(actual_returns.index)
        predictions = primary_predictions.loc[common_idx]
        returns = actual_returns.loc[common_idx]
        
        # Check if prediction was correct
        meta_labels = []
        for pred, ret in zip(predictions, returns):
            if pd.isna(pred) or pd.isna(ret):
                meta_labels.append(0)
                continue
            
            # Correct if sign matches and exceeds threshold
            if pred > 0:
                meta_labels.append(1 if ret > threshold else 0)
            else:
                meta_labels.append(1 if ret < -threshold else 0)
        
        return pd.Series(meta_labels, index=common_idx, name='meta_label')


class FeatureTransformer:
    """
    Advanced feature transformations for ML models.
    """
    
    @staticmethod
    def fractional_differentiation(
        series: pd.Series,
        d: float = 0.5,
        threshold: float = 1e-5
    ) -> pd.Series:
        """
        Apply fractional differentiation to achieve stationarity while preserving memory.
        
        Traditional differencing (d=1) removes all memory.
        Fractional differentiation (0 < d < 1) balances stationarity and memory.
        
        Args:
            series: Time series to differentiate
            d: Differentiation order (0.3-0.7 typical)
            threshold: Weight threshold for computational efficiency
        
        Returns:
            Fractionally differentiated series
        """
        # Compute weights using binomial coefficients
        weights = [1.0]
        k = 1
        while abs(weights[-1]) > threshold and k < 100:
            weight = -weights[-1] * (d - k + 1) / k
            weights.append(weight)
            k += 1
        
        weights = np.array(weights)
        w_len = len(weights)
        
        # Apply convolution (more efficient)
        result = np.full(len(series), np.nan)
        
        for i in range(w_len - 1, len(series)):
            # Dot product of weights with windowed series
            window = series.iloc[i - w_len + 1:i + 1].values[::-1]
            result[i] = np.dot(weights, window)
        
        return pd.Series(result, index=series.index)
    
    @staticmethod
    def time_decay_features(
        features: pd.DataFrame,
        half_life: int = 50
    ) -> pd.DataFrame:
        """
        Apply exponential time decay to features.
        
        Recent observations get higher weights than older ones.
        Useful for non-stationary environments.
        
        Args:
            features: Feature DataFrame
            half_life: Half-life in periods (50 = decay 50% after 50 bars)
        
        Returns:
            Time-decayed features
        """
        decay_rate = np.log(2) / half_life
        n = len(features)
        
        # Calculate weights (recent = 1.0, older exponentially decays)
        weights = np.exp(-decay_rate * np.arange(n-1, -1, -1))
        weights = weights / weights.sum()  # Normalize
        
        # Apply weights
        weighted_features = features.multiply(weights, axis=0)
        
        return weighted_features
    
    @staticmethod
    def create_target_returns(
        prices: pd.Series,
        horizon: int = 1,
        method: str = 'simple'
    ) -> pd.Series:
        """
        Create target returns for different prediction horizons.
        
        Args:
            prices: Price series
            horizon: Periods ahead to predict
            method: 'simple' or 'log' returns
        
        Returns:
            Target returns series
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(horizon))
        else:
            returns = (prices - prices.shift(horizon)) / prices.shift(horizon)
        
        # Shift back to align with features
        return returns.shift(-horizon)
    
    @staticmethod
    def volume_weighted_features(
        features: pd.DataFrame,
        volume: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Weight features by volume for better signal quality.
        
        Args:
            features: Feature DataFrame
            volume: Volume series
            window: Rolling window for volume normalization
        
        Returns:
            Volume-weighted features
        """
        # Normalize volume to [0, 1] range over rolling window
        volume_norm = volume / volume.rolling(window=window).max()
        volume_norm = volume_norm.fillna(0.5)  # Default to 0.5 if no data
        
        # Apply volume weighting
        weighted = features.multiply(volume_norm, axis=0)
        
        return weighted
