"""
Advanced ensemble models combining sentiment, predictions, anomalies, and RL.

Implements:
- Multi-model ensemble for signal generation
- Weighted voting based on model confidence
- Signal fusion and aggregation
- Ensemble performance evaluation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod


class SignalType(Enum):
    """Types of trading signals."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class ModelSignal:
    """Signal from an individual model."""
    model_name: str
    signal_type: SignalType
    confidence: float  # 0-1
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'signal': self.signal_type.name,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class EnsembleSignal:
    """Aggregated signal from ensemble."""
    final_signal: SignalType
    consensus_score: float  # -1 to 1
    confidence: float  # 0-1
    component_signals: List[ModelSignal]
    weights_used: Dict[str, float]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signal': self.final_signal.name,
            'consensus_score': self.consensus_score,
            'confidence': self.confidence,
            'num_models': len(self.component_signals),
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
        }


class SignalGenerator(ABC):
    """Base class for signal generators."""
    
    @abstractmethod
    def generate_signal(self, *args, **kwargs) -> ModelSignal:
        """Generate a trading signal."""
        pass


class SentimentSignalGenerator(SignalGenerator):
    """Generates signals based on sentiment analysis."""
    
    def __init__(self, recent_window: int = 10):
        """
        Initialize sentiment signal generator.
        
        Args:
            recent_window: Number of recent sentiments to consider
        """
        self.recent_window = recent_window
        self.recent_sentiments: List[float] = []
    
    def add_sentiment(self, sentiment_score: float):
        """Add a sentiment score (polarity, -1 to 1)."""
        self.recent_sentiments.append(sentiment_score)
        if len(self.recent_sentiments) > self.recent_window:
            self.recent_sentiments.pop(0)
    
    def generate_signal(self) -> ModelSignal:
        """Generate signal from sentiment data."""
        if not self.recent_sentiments:
            return ModelSignal(
                model_name="Sentiment",
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="No sentiment data available",
            )
        
        avg_sentiment = np.mean(self.recent_sentiments)
        sentiment_trend = (
            np.mean(self.recent_sentiments[-3:]) - np.mean(self.recent_sentiments[:3])
        ) if len(self.recent_sentiments) >= 6 else 0
        
        # Determine signal
        if avg_sentiment > 0.5 and sentiment_trend > 0.1:
            signal_type = SignalType.STRONG_BUY
            confidence = min(0.95, 0.5 + abs(avg_sentiment))
        elif avg_sentiment > 0.2:
            signal_type = SignalType.BUY
            confidence = min(0.85, 0.3 + abs(avg_sentiment))
        elif avg_sentiment < -0.5 and sentiment_trend < -0.1:
            signal_type = SignalType.STRONG_SELL
            confidence = min(0.95, 0.5 + abs(avg_sentiment))
        elif avg_sentiment < -0.2:
            signal_type = SignalType.SELL
            confidence = min(0.85, 0.3 + abs(avg_sentiment))
        else:
            signal_type = SignalType.HOLD
            confidence = min(0.5, 0.3)
        
        reasoning = f"Sentiment: {avg_sentiment:.2f}, Trend: {sentiment_trend:.2f}"
        
        return ModelSignal(
            model_name="Sentiment",
            signal_type=signal_type,
            confidence=confidence,
            reasoning=reasoning,
        )


class PredictionSignalGenerator(SignalGenerator):
    """Generates signals based on price predictions."""
    
    def __init__(self, prediction_horizon: int = 5):
        """
        Initialize prediction signal generator.
        
        Args:
            prediction_horizon: Days ahead to predict
        """
        self.prediction_horizon = prediction_horizon
        self.current_price = None
        self.predicted_price = None
        self.confidence = 0.0
    
    def set_prediction(self, current_price: float, predicted_price: float, confidence: float = 0.5):
        """Set prediction values."""
        self.current_price = current_price
        self.predicted_price = predicted_price
        self.confidence = confidence
    
    def generate_signal(self) -> ModelSignal:
        """Generate signal from prediction."""
        if self.current_price is None or self.predicted_price is None:
            return ModelSignal(
                model_name="Prediction",
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="No prediction available",
            )
        
        expected_return = (self.predicted_price - self.current_price) / self.current_price
        
        if expected_return > 0.10:
            signal_type = SignalType.STRONG_BUY
            conf = min(0.95, 0.5 + abs(expected_return) * 2)
        elif expected_return > 0.03:
            signal_type = SignalType.BUY
            conf = min(0.85, 0.3 + abs(expected_return) * 2)
        elif expected_return < -0.10:
            signal_type = SignalType.STRONG_SELL
            conf = min(0.95, 0.5 + abs(expected_return) * 2)
        elif expected_return < -0.03:
            signal_type = SignalType.SELL
            conf = min(0.85, 0.3 + abs(expected_return) * 2)
        else:
            signal_type = SignalType.HOLD
            conf = self.confidence
        
        reasoning = f"Expected return: {expected_return*100:.2f}%"
        
        return ModelSignal(
            model_name="Prediction",
            signal_type=signal_type,
            confidence=conf,
            reasoning=reasoning,
        )


class AnomalySignalGenerator(SignalGenerator):
    """Generates signals based on anomaly detection."""
    
    def __init__(self):
        """Initialize anomaly signal generator."""
        self.is_anomaly = False
        self.anomaly_severity = "low"
    
    def set_anomaly(self, is_anomaly: bool, severity: str = "low"):
        """Set anomaly detection results."""
        self.is_anomaly = is_anomaly
        self.anomaly_severity = severity
    
    def generate_signal(self) -> ModelSignal:
        """Generate signal from anomaly detection."""
        if not self.is_anomaly:
            return ModelSignal(
                model_name="Anomaly",
                signal_type=SignalType.HOLD,
                confidence=0.3,
                reasoning="No anomaly detected",
            )
        
        # Anomalies suggest caution
        if self.anomaly_severity == "critical":
            signal_type = SignalType.STRONG_SELL
            confidence = 0.9
        elif self.anomaly_severity == "high":
            signal_type = SignalType.SELL
            confidence = 0.8
        elif self.anomaly_severity == "medium":
            signal_type = SignalType.HOLD
            confidence = 0.5
        else:
            signal_type = SignalType.HOLD
            confidence = 0.3
        
        reasoning = f"Anomaly detected - severity: {self.anomaly_severity}"
        
        return ModelSignal(
            model_name="Anomaly",
            signal_type=signal_type,
            confidence=confidence,
            reasoning=reasoning,
        )


class RLSignalGenerator(SignalGenerator):
    """Generates signals based on RL agent decisions."""
    
    def __init__(self):
        """Initialize RL signal generator."""
        self.rl_action = None
        self.q_value = 0.0
    
    def set_action(self, action: str, q_value: float = 0.0):
        """Set RL action (BUY, SELL, HOLD)."""
        self.rl_action = action
        self.q_value = q_value
    
    def generate_signal(self) -> ModelSignal:
        """Generate signal from RL agent."""
        if self.rl_action is None:
            return ModelSignal(
                model_name="RL",
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reasoning="RL agent not initialized",
            )
        
        # Map RL action to signal
        if self.rl_action == "BUY":
            signal_type = SignalType.BUY
            confidence = min(0.9, 0.2 + abs(self.q_value) * 0.1)
        elif self.rl_action == "SELL":
            signal_type = SignalType.SELL
            confidence = min(0.9, 0.2 + abs(self.q_value) * 0.1)
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
        
        reasoning = f"RL action: {self.rl_action}, Q-value: {self.q_value:.3f}"
        
        return ModelSignal(
            model_name="RL",
            signal_type=signal_type,
            confidence=confidence,
            reasoning=reasoning,
        )


class EnsembleModel:
    """Combines multiple signal generators into ensemble."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        voting_method: str = "weighted",
        min_agreement: float = 0.3,
    ):
        """
        Initialize ensemble.
        
        Args:
            weights: Model weights (default: equal)
            voting_method: 'weighted', 'majority', 'unanimous'
            min_agreement: Minimum agreement for confidence
        """
        self.signal_generators: Dict[str, SignalGenerator] = {}
        self.voting_method = voting_method
        self.min_agreement = min_agreement
        
        self.weights = weights or {}
        self.signal_history: List[EnsembleSignal] = []
    
    def add_signal_generator(self, generator: SignalGenerator, weight: float = 1.0):
        """Add a signal generator to ensemble."""
        # Get model name from generator
        dummy_signal = generator.generate_signal()
        model_name = dummy_signal.model_name
        
        self.signal_generators[model_name] = generator
        self.weights[model_name] = weight
    
    def normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def generate_ensemble_signal(self, verbose: bool = False) -> EnsembleSignal:
        """Generate combined ensemble signal."""
        # Get signals from all generators
        component_signals = []
        for model_name, generator in self.signal_generators.items():
            signal = generator.generate_signal()
            component_signals.append(signal)
        
        if not component_signals:
            return EnsembleSignal(
                final_signal=SignalType.HOLD,
                consensus_score=0.0,
                confidence=0.0,
                component_signals=[],
                weights_used={},
                reasoning="No signal generators configured",
            )
        
        # Normalize weights
        self.normalize_weights()
        
        # Calculate weighted consensus score
        consensus_score = 0.0
        total_confidence = 0.0
        
        for signal in component_signals:
            weight = self.weights.get(signal.model_name, 1.0)
            consensus_score += signal.signal_type.value * weight * signal.confidence
            total_confidence += weight * signal.confidence
        
        # Normalize consensus score
        avg_confidence = total_confidence / len(component_signals) if component_signals else 0.0
        
        if total_confidence > 0:
            consensus_score = consensus_score / total_confidence
        
        # Determine final signal based on consensus score
        if consensus_score > 0.5:
            final_signal = SignalType.STRONG_BUY if consensus_score > 1.0 else SignalType.BUY
        elif consensus_score < -0.5:
            final_signal = SignalType.STRONG_SELL if consensus_score < -1.0 else SignalType.SELL
        else:
            final_signal = SignalType.HOLD
        
        # Build reasoning
        reasoning_parts = [f"{s.model_name}: {s.reasoning}" for s in component_signals]
        reasoning = " | ".join(reasoning_parts)
        
        ensemble_signal = EnsembleSignal(
            final_signal=final_signal,
            consensus_score=consensus_score,
            confidence=min(0.99, avg_confidence),
            component_signals=component_signals,
            weights_used=self.weights.copy(),
            reasoning=reasoning,
        )
        
        self.signal_history.append(ensemble_signal)
        
        return ensemble_signal
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics on ensemble signals."""
        if not self.signal_history:
            return {}
        
        signals_array = np.array([s.final_signal.value for s in self.signal_history])
        consensus_array = np.array([s.consensus_score for s in self.signal_history])
        confidence_array = np.array([s.confidence for s in self.signal_history])
        
        return {
            'num_signals': len(self.signal_history),
            'mean_consensus': float(np.mean(consensus_array)),
            'mean_confidence': float(np.mean(confidence_array)),
            'buy_ratio': float(np.sum(signals_array > 0) / len(signals_array)),
            'sell_ratio': float(np.sum(signals_array < 0) / len(signals_array)),
            'hold_ratio': float(np.sum(signals_array == 0) / len(signals_array)),
        }


class EnsembleBacktester:
    """Backtest ensemble strategy on historical data."""
    
    @staticmethod
    def backtest(
        ensemble: EnsembleModel,
        price_series: pd.Series,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Backtest ensemble on price series.
        
        Args:
            ensemble: Ensemble model
            price_series: Historical prices
            initial_capital: Starting capital
            transaction_cost: Transaction cost ratio
            
        Returns:
            Backtest results dictionary
        """
        portfolio_value = initial_capital
        position = False  # Currently holding
        entry_price = 0.0
        trades = []
        portfolio_history = [initial_capital]
        
        for i in range(len(price_series)):
            current_price = price_series.iloc[i]
            
            signal = ensemble.generate_ensemble_signal()
            
            # Simple strategy: buy on strong buy, sell on strong sell
            if signal.final_signal in [SignalType.STRONG_BUY, SignalType.BUY] and not position:
                position = True
                entry_price = current_price
                trades.append(('buy', current_price, i))
            
            elif signal.final_signal in [SignalType.STRONG_SELL, SignalType.SELL] and position:
                pnl = (current_price - entry_price) * (portfolio_value / entry_price)
                portfolio_value = portfolio_value * (1 - transaction_cost) + pnl
                trades.append(('sell', current_price, i))
                position = False
            
            # Update portfolio value if holding
            if position:
                portfolio_value = initial_capital + (current_price - entry_price) * (initial_capital / entry_price) * (1 - transaction_cost)
            
            portfolio_history.append(portfolio_value)
        
        # Close position at end if open
        if position:
            final_price = price_series.iloc[-1]
            pnl = (final_price - entry_price) * (portfolio_value / entry_price)
            portfolio_value = portfolio_value * (1 - transaction_cost) + pnl
        
        # Calculate metrics
        portfolio_array = np.array(portfolio_history)
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        
        cummax = np.maximum.accumulate(portfolio_array)
        drawdown = (cummax - portfolio_array) / cummax
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'final_portfolio_value': portfolio_value,
            'trades': trades,
            'portfolio_history': portfolio_history,
        }
