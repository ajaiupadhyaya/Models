"""
Tests for advanced ensemble models.

Tests:
- Signal generators (sentiment, prediction, anomaly, RL)
- Ensemble voting mechanisms
- Signal aggregation and fusion
- Ensemble backtesting
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from core.ensemble_models import (
    SignalType,
    ModelSignal,
    EnsembleSignal,
    SentimentSignalGenerator,
    PredictionSignalGenerator,
    AnomalySignalGenerator,
    RLSignalGenerator,
    EnsembleModel,
    EnsembleBacktester,
)


class TestSignalType:
    """Test signal types."""
    
    def test_signal_values(self):
        """Test signal type values."""
        assert SignalType.STRONG_BUY.value == 2
        assert SignalType.BUY.value == 1
        assert SignalType.HOLD.value == 0
        assert SignalType.SELL.value == -1
        assert SignalType.STRONG_SELL.value == -2


class TestModelSignal:
    """Test model signal representation."""
    
    def test_signal_creation(self):
        """Test creating a model signal."""
        signal = ModelSignal(
            model_name="TestModel",
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning="Test reasoning",
        )
        
        assert signal.model_name == "TestModel"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = ModelSignal(
            model_name="TestModel",
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning="Test",
            timestamp=datetime(2024, 1, 1, 10, 0),
        )
        
        signal_dict = signal.to_dict()
        
        assert signal_dict['model_name'] == "TestModel"
        assert signal_dict['signal'] == "BUY"
        assert signal_dict['confidence'] == 0.8


class TestEnsembleSignal:
    """Test ensemble signal."""
    
    def test_ensemble_signal_creation(self):
        """Test creating ensemble signal."""
        component_signals = [
            ModelSignal("Model1", SignalType.BUY, 0.8, "Reason1"),
            ModelSignal("Model2", SignalType.BUY, 0.7, "Reason2"),
        ]
        
        ensemble_sig = EnsembleSignal(
            final_signal=SignalType.BUY,
            consensus_score=0.75,
            confidence=0.75,
            component_signals=component_signals,
            weights_used={"Model1": 0.5, "Model2": 0.5},
            reasoning="Combined signal",
        )
        
        assert ensemble_sig.final_signal == SignalType.BUY
        assert ensemble_sig.consensus_score == 0.75
        assert len(ensemble_sig.component_signals) == 2
    
    def test_ensemble_signal_to_dict(self):
        """Test ensemble signal serialization."""
        ensemble_sig = EnsembleSignal(
            final_signal=SignalType.BUY,
            consensus_score=0.75,
            confidence=0.75,
            component_signals=[],
            weights_used={},
            reasoning="Test",
        )
        
        sig_dict = ensemble_sig.to_dict()
        
        assert sig_dict['signal'] == "BUY"
        assert sig_dict['consensus_score'] == 0.75


class TestSentimentSignalGenerator:
    """Test sentiment signal generation."""
    
    @pytest.fixture
    def sentiment_gen(self):
        """Create sentiment generator."""
        return SentimentSignalGenerator()
    
    def test_initialization(self, sentiment_gen):
        """Test sentiment generator initialization."""
        assert sentiment_gen.recent_window == 10
        assert len(sentiment_gen.recent_sentiments) == 0
    
    def test_add_sentiment(self, sentiment_gen):
        """Test adding sentiment values."""
        sentiment_gen.add_sentiment(0.5)
        sentiment_gen.add_sentiment(0.6)
        
        assert len(sentiment_gen.recent_sentiments) == 2
        assert sentiment_gen.recent_sentiments[0] == 0.5
    
    def test_window_overflow(self, sentiment_gen):
        """Test that window doesn't overflow."""
        for i in range(15):
            sentiment_gen.add_sentiment(0.5)
        
        assert len(sentiment_gen.recent_sentiments) == 10
    
    def test_strong_positive_sentiment(self, sentiment_gen):
        """Test signal generation with strong positive sentiment."""
        for _ in range(5):
            sentiment_gen.add_sentiment(0.7)
        
        signal = sentiment_gen.generate_signal()
        
        assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence > 0.5
    
    def test_strong_negative_sentiment(self, sentiment_gen):
        """Test signal generation with strong negative sentiment."""
        for _ in range(5):
            sentiment_gen.add_sentiment(-0.7)
        
        signal = sentiment_gen.generate_signal()
        
        assert signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.confidence > 0.5
    
    def test_neutral_sentiment(self, sentiment_gen):
        """Test signal generation with neutral sentiment."""
        sentiment_gen.add_sentiment(0.0)
        signal = sentiment_gen.generate_signal()
        
        assert signal.signal_type == SignalType.HOLD


class TestPredictionSignalGenerator:
    """Test prediction signal generation."""
    
    @pytest.fixture
    def prediction_gen(self):
        """Create prediction generator."""
        return PredictionSignalGenerator()
    
    def test_initialization(self, prediction_gen):
        """Test prediction generator initialization."""
        assert prediction_gen.current_price is None
        assert prediction_gen.predicted_price is None
    
    def test_strong_upside_prediction(self, prediction_gen):
        """Test signal on strong upside prediction."""
        prediction_gen.set_prediction(100.0, 112.0, confidence=0.8)
        
        signal = prediction_gen.generate_signal()
        
        assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence > 0.5
    
    def test_strong_downside_prediction(self, prediction_gen):
        """Test signal on strong downside prediction."""
        prediction_gen.set_prediction(100.0, 88.0, confidence=0.8)
        
        signal = prediction_gen.generate_signal()
        
        assert signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.confidence > 0.5
    
    def test_neutral_prediction(self, prediction_gen):
        """Test signal on neutral prediction."""
        prediction_gen.set_prediction(100.0, 101.0, confidence=0.8)
        
        signal = prediction_gen.generate_signal()
        
        assert signal.signal_type == SignalType.HOLD


class TestAnomalySignalGenerator:
    """Test anomaly signal generation."""
    
    @pytest.fixture
    def anomaly_gen(self):
        """Create anomaly generator."""
        return AnomalySignalGenerator()
    
    def test_no_anomaly(self, anomaly_gen):
        """Test signal with no anomaly."""
        anomaly_gen.set_anomaly(False)
        
        signal = anomaly_gen.generate_signal()
        
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.3
    
    def test_critical_anomaly(self, anomaly_gen):
        """Test signal with critical anomaly."""
        anomaly_gen.set_anomaly(True, severity="critical")
        
        signal = anomaly_gen.generate_signal()
        
        assert signal.signal_type == SignalType.STRONG_SELL
        assert signal.confidence == 0.9
    
    def test_high_severity_anomaly(self, anomaly_gen):
        """Test signal with high severity anomaly."""
        anomaly_gen.set_anomaly(True, severity="high")
        
        signal = anomaly_gen.generate_signal()
        
        assert signal.signal_type == SignalType.SELL


class TestRLSignalGenerator:
    """Test RL signal generation."""
    
    @pytest.fixture
    def rl_gen(self):
        """Create RL generator."""
        return RLSignalGenerator()
    
    def test_buy_action(self, rl_gen):
        """Test signal from BUY action."""
        rl_gen.set_action("BUY", q_value=0.5)
        
        signal = rl_gen.generate_signal()
        
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.2
    
    def test_sell_action(self, rl_gen):
        """Test signal from SELL action."""
        rl_gen.set_action("SELL", q_value=0.5)
        
        signal = rl_gen.generate_signal()
        
        assert signal.signal_type == SignalType.SELL
    
    def test_hold_action(self, rl_gen):
        """Test signal from HOLD action."""
        rl_gen.set_action("HOLD", q_value=0.0)
        
        signal = rl_gen.generate_signal()
        
        assert signal.signal_type == SignalType.HOLD


class TestEnsembleModel:
    """Test ensemble model."""
    
    @pytest.fixture
    def ensemble(self):
        """Create ensemble model."""
        return EnsembleModel()
    
    @pytest.fixture
    def ensemble_with_generators(self):
        """Create ensemble with signal generators."""
        ensemble = EnsembleModel()
        ensemble.add_signal_generator(SentimentSignalGenerator(), weight=0.3)
        ensemble.add_signal_generator(PredictionSignalGenerator(), weight=0.4)
        ensemble.add_signal_generator(AnomalySignalGenerator(), weight=0.3)
        return ensemble
    
    def test_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert len(ensemble.signal_generators) == 0
        assert ensemble.voting_method == "weighted"
    
    def test_add_generator(self, ensemble):
        """Test adding signal generator."""
        sentiment_gen = SentimentSignalGenerator()
        ensemble.add_signal_generator(sentiment_gen, weight=1.0)
        
        assert "Sentiment" in ensemble.signal_generators
    
    def test_normalize_weights(self, ensemble_with_generators):
        """Test weight normalization."""
        ensemble_with_generators.normalize_weights()
        
        total_weight = sum(ensemble_with_generators.weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def test_generate_ensemble_signal_empty(self, ensemble):
        """Test ensemble signal with no generators."""
        signal = ensemble.generate_ensemble_signal()
        
        assert signal.final_signal == SignalType.HOLD
        assert signal.confidence == 0.0
    
    def test_generate_ensemble_signal_bullish(self, ensemble_with_generators):
        """Test ensemble signal with bullish components."""
        # Set up bullish signals
        for gen in ensemble_with_generators.signal_generators.values():
            if isinstance(gen, SentimentSignalGenerator):
                gen.add_sentiment(0.7)
            elif isinstance(gen, PredictionSignalGenerator):
                gen.set_prediction(100.0, 110.0)
        
        signal = ensemble_with_generators.generate_ensemble_signal()
        
        assert signal.final_signal in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.consensus_score > 0
    
    def test_generate_ensemble_signal_bearish(self, ensemble_with_generators):
        """Test ensemble signal with bearish components."""
        # Set up bearish signals
        for gen in ensemble_with_generators.signal_generators.values():
            if isinstance(gen, SentimentSignalGenerator):
                gen.add_sentiment(-0.7)
            elif isinstance(gen, PredictionSignalGenerator):
                gen.set_prediction(100.0, 85.0)
        
        signal = ensemble_with_generators.generate_ensemble_signal()
        
        assert signal.final_signal in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.consensus_score < 0
    
    def test_consensus_disagreement(self, ensemble_with_generators):
        """Test ensemble when models disagree."""
        sentiment_gen = ensemble_with_generators.signal_generators.get("Sentiment")
        pred_gen = ensemble_with_generators.signal_generators.get("Prediction")
        
        if sentiment_gen:
            sentiment_gen.add_sentiment(0.7)
        if pred_gen:
            pred_gen.set_prediction(100.0, 95.0)
        
        signal = ensemble_with_generators.generate_ensemble_signal()
        
        # Should be mixed signal
        assert signal.final_signal == SignalType.HOLD
    
    def test_signal_history(self, ensemble_with_generators):
        """Test that signal history is tracked."""
        ensemble_with_generators.generate_ensemble_signal()
        ensemble_with_generators.generate_ensemble_signal()
        
        assert len(ensemble_with_generators.signal_history) == 2
    
    def test_get_signal_statistics(self, ensemble_with_generators):
        """Test getting signal statistics."""
        # Generate multiple signals
        for _ in range(5):
            ensemble_with_generators.generate_ensemble_signal()
        
        stats = ensemble_with_generators.get_signal_statistics()
        
        assert stats['num_signals'] == 5
        assert 'mean_consensus' in stats
        assert 'mean_confidence' in stats
        assert stats['buy_ratio'] + stats['sell_ratio'] + stats['hold_ratio'] > 0.99


class TestEnsembleIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def full_ensemble(self):
        """Create full ensemble."""
        ensemble = EnsembleModel()
        ensemble.add_signal_generator(SentimentSignalGenerator(), weight=0.25)
        ensemble.add_signal_generator(PredictionSignalGenerator(), weight=0.25)
        ensemble.add_signal_generator(AnomalySignalGenerator(), weight=0.25)
        ensemble.add_signal_generator(RLSignalGenerator(), weight=0.25)
        return ensemble
    
    def test_all_generators_together(self, full_ensemble):
        """Test ensemble with all signal types."""
        # Setup different signals
        sentiment_gen = full_ensemble.signal_generators["Sentiment"]
        sentiment_gen.add_sentiment(0.5)
        
        pred_gen = full_ensemble.signal_generators["Prediction"]
        pred_gen.set_prediction(100.0, 105.0)
        
        anomaly_gen = full_ensemble.signal_generators["Anomaly"]
        anomaly_gen.set_anomaly(False)
        
        rl_gen = full_ensemble.signal_generators["RL"]
        rl_gen.set_action("BUY")
        
        signal = full_ensemble.generate_ensemble_signal()
        
        # Should produce moderate bullish signal
        assert signal.final_signal in [SignalType.BUY, SignalType.HOLD]
        assert len(signal.component_signals) == 4


class TestEnsembleBacktester:
    """Test ensemble backtesting."""
    
    @pytest.fixture
    def price_series(self):
        """Generate sample price series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        # Trending up
        prices = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def ensemble(self):
        """Create simple ensemble."""
        ensemble = EnsembleModel()
        ensemble.add_signal_generator(SentimentSignalGenerator())
        ensemble.add_signal_generator(PredictionSignalGenerator())
        return ensemble
    
    def test_backtest_creation(self, ensemble, price_series):
        """Test backtest runs without error."""
        results = EnsembleBacktester.backtest(ensemble, price_series)
        
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'num_trades' in results
    
    def test_backtest_metrics(self, ensemble, price_series):
        """Test backtest metrics are calculated."""
        results = EnsembleBacktester.backtest(ensemble, price_series)
        
        assert isinstance(results['total_return'], (int, float))
        assert isinstance(results['sharpe_ratio'], (int, float))
        assert isinstance(results['max_drawdown'], (int, float))
        assert results['max_drawdown'] >= 0
    
    def test_backtest_portfolio_history(self, ensemble, price_series):
        """Test portfolio history tracking."""
        results = EnsembleBacktester.backtest(ensemble, price_series)
        
        assert len(results['portfolio_history']) >= len(price_series)
        # Portfolio should start with initial capital
        assert abs(results['portfolio_history'][0] - 10000.0) < 0.01
    
    def test_backtest_with_different_capital(self, ensemble, price_series):
        """Test backtest with different initial capital."""
        results = EnsembleBacktester.backtest(
            ensemble, price_series, initial_capital=50000.0
        )
        
        # Should scale results proportionally
        assert isinstance(results['final_portfolio_value'], (int, float))


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_ensemble(self):
        """Test ensemble with no generators."""
        ensemble = EnsembleModel()
        signal = ensemble.generate_ensemble_signal()
        
        assert signal.confidence == 0.0
    
    def test_conflicting_signals(self):
        """Test ensemble with conflicting signals."""
        ensemble = EnsembleModel()
        
        sentiment_gen = SentimentSignalGenerator()
        sentiment_gen.add_sentiment(0.5)  # Moderate bullish
        ensemble.add_signal_generator(sentiment_gen, weight=0.5)
        
        pred_gen = PredictionSignalGenerator()
        pred_gen.set_prediction(100.0, 98.0, confidence=0.5)  # Weak bearish
        ensemble.add_signal_generator(pred_gen, weight=0.5)
        
        signal = ensemble.generate_ensemble_signal()
        
        # With moderate signals, should be close to neutral
        assert signal.final_signal in [SignalType.HOLD, SignalType.BUY, SignalType.SELL]
    
    def test_single_high_confidence_signal(self):
        """Test ensemble dominated by single high-confidence signal."""
        ensemble = EnsembleModel()
        
        pred_gen = PredictionSignalGenerator()
        pred_gen.set_prediction(100.0, 125.0, confidence=0.99)
        ensemble.add_signal_generator(pred_gen, weight=0.9)
        
        sentiment_gen = SentimentSignalGenerator()
        sentiment_gen.add_sentiment(-0.5)  # Weak bearish
        ensemble.add_signal_generator(sentiment_gen, weight=0.1)
        
        signal = ensemble.generate_ensemble_signal()
        
        # Should dominate with prediction signal
        assert signal.final_signal in [SignalType.BUY, SignalType.STRONG_BUY]
