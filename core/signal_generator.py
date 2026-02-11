"""
Advanced Signal Generation and Market Regime Detection
Intelligent signal filtering with multi-factor confirmation and regime awareness.

Features:
- Multi-factor signal confirmation
- Market regime detection (trending, mean-reverting, volatile)
- Signal strength and confidence calculation
- Smart entry and exit signals
- Risk-adjusted position sizing
- Regime-adaptive strategy adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class SignalType(Enum):
    """Types of trading signals."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal with metadata."""
    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    factors: Dict[str, float]  # Contributing factors and their scores
    regime: MarketRegime = MarketRegime.NEUTRAL
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 1.0
    risk_reward_ratio: float = 0.0
    reason: str = ""
    
    def is_valid(self) -> bool:
        """Check if signal is valid."""
        return (
            self.strength > 0.5 and 
            self.confidence > 0.5 and
            self.signal_type != SignalType.HOLD
        )
    
    def get_quality_score(self) -> float:
        """Get overall signal quality (0 to 1)."""
        # Weighted combination of strength and confidence
        return 0.6 * self.strength + 0.4 * self.confidence


class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators.
    """
    
    def __init__(self, lookback: int = 50):
        """
        Initialize detector.
        
        Args:
            lookback: Lookback period for calculations
        """
        self.lookback = lookback
        self.regime_cache: Dict[str, MarketRegime] = {}
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect market regime from price data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            MarketRegime enum
        """
        if df.empty or len(df) < self.lookback:
            return MarketRegime.NEUTRAL
        
        # Get recent data
        prices = df['Close'].iloc[-self.lookback:]
        returns = prices.pct_change().dropna()
        
        # Calculate metrics
        trend_strength = self._calculate_trend_strength(prices)
        volatility = returns.std()
        volatility_z_score = self._get_volatility_zscore(df)
        
        # Detect regime
        if volatility_z_score > 2.0:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility_z_score < -1.0:
            return MarketRegime.LOW_VOLATILITY
        
        if trend_strength > 0.7:
            return MarketRegime.STRONG_UPTREND
        elif trend_strength > 0.3:
            return MarketRegime.UPTREND
        elif trend_strength < -0.7:
            return MarketRegime.STRONG_DOWNTREND
        elif trend_strength < -0.3:
            return MarketRegime.DOWNTREND
        
        return MarketRegime.NEUTRAL
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """
        Calculate trend strength using linear regression.
        
        Returns:
            Float between -1 and 1
        """
        x = np.arange(len(prices))
        y = prices.values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Normalize slope to -1 to 1
        price_range = y.max() - y.min()
        if price_range == 0:
            return 0.0
        
        normalized_slope = slope / (price_range / len(prices))
        return np.clip(normalized_slope, -1, 1)
    
    def _get_volatility_zscore(self, df: pd.DataFrame) -> float:
        """Get z-score of volatility."""
        if len(df) < self.lookback:
            return 0.0
        
        returns = df['Close'].pct_change().dropna()
        
        # Recent volatility
        recent_vol = returns.iloc[-20:].std()
        
        # Historical volatility
        historical_vol = returns.std()
        
        if historical_vol == 0:
            return 0.0
        
        z_score = (recent_vol - historical_vol) / (historical_vol / np.sqrt(20))
        return z_score
    
    def is_trending(self, df: pd.DataFrame) -> bool:
        """Check if market is in trending regime."""
        regime = self.detect_regime(df)
        return regime in [
            MarketRegime.STRONG_UPTREND,
            MarketRegime.UPTREND,
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.DOWNTREND
        ]
    
    def is_volatile(self, df: pd.DataFrame) -> bool:
        """Check if market is in high volatility regime."""
        regime = self.detect_regime(df)
        return regime == MarketRegime.HIGH_VOLATILITY


class AdvancedSignalGenerator:
    """
    Generates high-quality trading signals using multiple factors.
    """
    
    def __init__(self, lookback: int = 50):
        """
        Initialize signal generator.
        
        Args:
            lookback: Lookback period for calculations
        """
        self.lookback = lookback
        self.regime_detector = MarketRegimeDetector(lookback)
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from price data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            List of Signal objects
        """
        if df.empty or len(df) < self.lookback:
            return []
        
        latest_idx = df.index[-1]
        latest_price = df['Close'].iloc[-1]
        
        signals = []
        
        # Get regime
        regime = self.regime_detector.detect_regime(df)
        
        # Calculate all factors
        factors = self._calculate_all_factors(df)
        
        # Determine signal
        signal_type, strength = self._determine_signal_type(df, factors, regime)
        confidence = self._calculate_confidence(factors)
        
        if signal_type != SignalType.HOLD:
            # Calculate risk management levels
            stop_loss, take_profit = self._calculate_risk_levels(
                latest_price, signal_type, factors, regime
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                strength, confidence, factors
            )
            
            # Calculate risk/reward
            risk_reward = self._calculate_risk_reward(
                latest_price, stop_loss, take_profit, signal_type
            )
            
            signal = Signal(
                timestamp=latest_idx,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                factors=factors,
                regime=regime,
                entry_price=latest_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_reward_ratio=risk_reward,
                reason=self._generate_signal_reason(factors, regime)
            )
            
            if signal.is_valid():
                signals.append(signal)
        
        return signals
    
    def _calculate_all_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all signal factors."""
        factors = {}
        
        # Momentum factors
        factors['rsi'] = self._calculate_rsi(df)
        factors['macd_signal'] = self._calculate_macd_signal(df)
        factors['momentum'] = self._calculate_momentum(df)
        
        # Mean reversion factors
        factors['mean_reversion'] = self._calculate_mean_reversion(df)
        factors['bollinger_signal'] = self._calculate_bollinger_signal(df)
        
        # Trend factors
        factors['trend_strength'] = self.regime_detector._calculate_trend_strength(df['Close'])
        factors['moving_average_alignment'] = self._calculate_ma_alignment(df)
        
        # Volume factors
        factors['volume_signal'] = self._calculate_volume_signal(df)
        
        # Volatility factors
        factors['volatility_regime'] = self._get_volatility_score(df)
        
        return factors
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI signal (-1 to 1)."""
        if len(df) < period + 1:
            return 0.0
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if rs.iloc[-1] != 0 else 50
        
        # Normalize to -1 to 1: sell at 70, buy at 30
        signal = 2 * (rsi - 50) / 100
        return np.clip(signal, -1, 1)
    
    def _calculate_macd_signal(self, df: pd.DataFrame) -> float:
        """Calculate MACD signal (-1 to 1)."""
        if len(df) < 26:
            return 0.0
        
        prices = df['Close']
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal_line
        
        # Signal: positive histogram = bullish
        strength = np.clip(histogram.iloc[-1] / df['Close'].std(), -1, 1)
        return strength
    
    def _calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> float:
        """Calculate price momentum (-1 to 1)."""
        if len(df) < period:
            return 0.0
        
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-period]) / df['Close'].iloc[-period]
        return np.clip(price_change * 10, -1, 1)  # Normalize
    
    def _calculate_mean_reversion(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate mean reversion signal (-1 to 1)."""
        if len(df) < period:
            return 0.0
        
        prices = df['Close'].iloc[-period:]
        mean = prices.mean()
        current = prices.iloc[-1]
        std = prices.std()
        
        if std == 0:
            return 0.0
        
        # Z-score: positive = overextended high (sell), negative = overextended low (buy)
        z_score = (current - mean) / std
        return np.clip(z_score * 0.5, -1, 1)  # Dampen
    
    def _calculate_bollinger_signal(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Bollinger Band signal (-1 to 1)."""
        if len(df) < period:
            return 0.0
        
        prices = df['Close'].iloc[-period:]
        sma = prices.mean()
        std = prices.std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        current = prices.iloc[-1]
        
        if upper_band == lower_band:
            return 0.0
        
        # Normalize position within bands
        position = (current - lower_band) / (upper_band - lower_band)
        # Signal: 0.5 = middle, <0.2 = oversold (buy), >0.8 = overbought (sell)
        return np.clip((0.5 - position) * 2, -1, 1)
    
    def _calculate_ma_alignment(self, df: pd.DataFrame) -> float:
        """Calculate moving average alignment signal."""
        if len(df) < 50:
            return 0.0
        
        prices = df['Close']
        sma20 = prices.rolling(20).mean()
        sma50 = prices.rolling(50).mean()
        current = prices.iloc[-1]
        
        if current < sma20.iloc[-1]:
            return -1.0
        elif current > sma50.iloc[-1]:
            return 1.0
        else:
            return 0.0
    
    def _calculate_volume_signal(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate volume signal (-1 to 1)."""
        if len(df) < period:
            return 0.0
        
        recent_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].iloc[-period:].mean()
        
        if avg_vol == 0:
            return 0.0
        
        vol_ratio = recent_vol / avg_vol
        # Signal: high volume on up day = bullish, high volume on down day = bearish
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        direction = 1.0 if price_change > 0 else -1.0
        
        return np.clip(direction * (vol_ratio - 1) * 0.5, -1, 1)
    
    def _get_volatility_score(self, df: pd.DataFrame) -> float:
        """Get volatility regime score (-1 = low, 1 = high)."""
        volatility_zscore = self.regime_detector._get_volatility_zscore(df)
        return np.clip(volatility_zscore / 3, -1, 1)
    
    def _determine_signal_type(self, df: pd.DataFrame, factors: Dict[str, float], 
                              regime: MarketRegime) -> Tuple[SignalType, float]:
        """Determine signal type and strength."""
        # Weighted combination of factors
        weights = {
            'momentum': 0.25,
            'rsi': 0.20,
            'macd_signal': 0.20,
            'mean_reversion': 0.15,
            'trend_strength': 0.10,
            'moving_average_alignment': 0.10,
        }
        
        score = sum(factors.get(k, 0) * w for k, w in weights.items())
        
        # Regime adjustment
        if regime == MarketRegime.HIGH_VOLATILITY:
            score *= 0.8  # Reduce signal strength in high volatility
        elif regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            score *= 1.2  # Boost signal strength in strong trends
        
        strength = np.clip(abs(score), 0, 1)
        
        # Determine signal type
        if score > 0.5:
            signal_type = SignalType.ENTRY_LONG
        elif score < -0.5:
            signal_type = SignalType.ENTRY_SHORT
        else:
            signal_type = SignalType.HOLD
        
        return signal_type, strength
    
    def _calculate_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate signal confidence based on factor agreement."""
        # Count factors agreeing with signal direction
        positive_factors = sum(1 for f in factors.values() if f > 0.3)
        negative_factors = sum(1 for f in factors.values() if f < -0.3)
        total_factors = len(factors)
        
        # Confidence based on factor alignment
        max_agreement = max(positive_factors, negative_factors)
        confidence = max_agreement / total_factors if total_factors > 0 else 0.5
        
        return np.clip(confidence, 0, 1)
    
    def _calculate_risk_levels(self, price: float, signal_type: SignalType,
                              factors: Dict[str, float], regime: MarketRegime
                              ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        volatility_multiplier = 1.0 + abs(factors.get('volatility_regime', 0))
        
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND]:
            atr_equivalent = price * 0.02 * volatility_multiplier
        else:
            atr_equivalent = price * 0.03 * volatility_multiplier
        
        if signal_type == SignalType.ENTRY_LONG:
            stop_loss = price - atr_equivalent
            take_profit = price + (atr_equivalent * 2)  # 1:2 risk:reward
        else:  # ENTRY_SHORT
            stop_loss = price + atr_equivalent
            take_profit = price - (atr_equivalent * 2)
        
        return stop_loss, take_profit
    
    def _calculate_position_size(self, strength: float, confidence: float,
                                factors: Dict[str, float]) -> float:
        """Calculate position size based on signal quality."""
        # Position size increases with signal quality
        quality = 0.6 * strength + 0.4 * confidence
        
        # Reduce in high volatility
        vol_adjustment = 1.0 - abs(factors.get('volatility_regime', 0)) * 0.3
        
        position_size = quality * vol_adjustment
        return np.clip(position_size, 0.1, 1.0)
    
    def _calculate_risk_reward(self, entry: float, stop_loss: float, 
                              take_profit: float, signal_type: SignalType) -> float:
        """Calculate risk/reward ratio."""
        if signal_type == SignalType.ENTRY_LONG:
            risk = entry - stop_loss
            reward = take_profit - entry
        else:
            risk = stop_loss - entry
            reward = entry - take_profit
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def _generate_signal_reason(self, factors: Dict[str, float], 
                               regime: MarketRegime) -> str:
        """Generate human-readable signal reason."""
        top_factors = sorted(
            factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]
        
        reasons = []
        for factor_name, factor_value in top_factors:
            if abs(factor_value) > 0.5:
                direction = "bullish" if factor_value > 0 else "bearish"
                reasons.append(f"{factor_name} {direction}")
        
        reason = f"Regime: {regime.value}. Signals: {', '.join(reasons)}"
        return reason
