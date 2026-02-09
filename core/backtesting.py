"""
Advanced Backtesting Engine with ML/DL/RL Support
Machine learning enhanced backtesting with multiple prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 1
    position_type: str = 'long'  # 'long' or 'short'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    def close(self, exit_date: datetime, exit_price: float):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.position_type == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price


class BacktestSignal:
    """
    Signal generation from predictions.
    """
    
    def __init__(self, timestamp: datetime, asset: str, signal: float, confidence: float = 0.5):
        """
        Initialize signal.
        
        Args:
            timestamp: Signal timestamp
            asset: Asset ticker
            signal: Signal value (-1 to 1, where 1 = strong buy, -1 = strong sell)
            confidence: Confidence level (0 to 1)
        """
        self.timestamp = timestamp
        self.asset = asset
        self.signal = np.clip(signal, -1, 1)
        self.confidence = np.clip(confidence, 0, 1)
    
    def get_position_type(self, threshold: float = 0.0) -> Optional[str]:
        """Get position type from signal."""
        if self.signal > threshold:
            return 'long'
        elif self.signal < -threshold:
            return 'short'
        return None
    
    def get_position_size(self, base_size: float = 1.0) -> float:
        """Get position size based on signal strength."""
        return abs(self.signal) * self.confidence * base_size


class SimpleMLPredictor:
    """
    Simple ML-based price predictor using technical indicators and momentum.
    Foundation for more complex models.
    """
    
    def __init__(self, lookback_window: int = 20):
        """
        Initialize predictor.
        
        Args:
            lookback_window: Number of periods for feature calculation
        """
        self.lookback_window = lookback_window
        self.model = None
        self.scaler = None
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicator features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Momentum
        features['momentum'] = df['Close'] - df['Close'].shift(self.lookback_window)
        features['momentum_pct'] = (df['Close'] - df['Close'].shift(self.lookback_window)) / df['Close'].shift(self.lookback_window)
        
        # Mean reversion (distance from moving average)
        features['sma_20'] = df['Close'].rolling(20).mean()
        features['sma_50'] = df['Close'].rolling(50).mean()
        features['price_to_sma20'] = df['Close'] / features['sma_20'] - 1
        features['price_to_sma50'] = df['Close'] / features['sma_50'] - 1
        
        # Volatility
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        features['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Volume features
        features['volume_change'] = df['Volume'].pct_change()
        features['price_volume_trend'] = (df['Close'].pct_change() * df['Volume']).rolling(20).mean()
        
        # RSI-like momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_like'] = 100 - (100 / (1 + rs))
        
        return features.dropna()
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict next period return direction.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Array of signals (-1 to 1)
        """
        features = self.calculate_features(df)
        
        if features.empty:
            return np.zeros(len(df))
        
        # Simple rules-based prediction (can be replaced with ML model)
        signals = np.zeros(len(features))
        
        for i in range(len(features)):
            score = 0
            
            # Momentum signal
            if features['momentum_pct'].iloc[i] > 0:
                score += 0.3
            else:
                score -= 0.3
            
            # Mean reversion
            if features['price_to_sma20'].iloc[i] < -0.02:
                score += 0.2  # Oversold, mean reversion expected
            elif features['price_to_sma20'].iloc[i] > 0.02:
                score -= 0.2
            
            # RSI
            rsi = features['rsi_like'].iloc[i]
            if rsi < 30:
                score += 0.2
            elif rsi > 70:
                score -= 0.2
            
            # Volume confirmation
            if features['volume_change'].iloc[i] > 0:
                score += 0.15
            else:
                score -= 0.15
            
            # Volatility adjust
            if features['volatility'].iloc[i] > features['volatility'].mean():
                score *= 0.8  # Reduce signal in high volatility
            
            signals[i] = np.clip(score, -1, 1)
        
        return signals


class BacktestEngine:
    """
    Complete backtesting engine with ML prediction support and robust validation.
    Integrated with TradingCalendar for realistic trading day filtering.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000, 
                 commission: float = 0.001,
                 exchange: Optional[str] = None):
        """
        Initialize backtesting engine with input validation.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (e.g., 0.001 = 0.1%)
            exchange: Optional exchange for trading calendar (NYSE, NASDAQ, etc.)
                     If provided, backtest only trades on actual trading days
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if commission < 0 or commission > 1:
            raise ValueError("Commission must be between 0 and 1")
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.capital = initial_capital
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        self.signals = []
        
        # Trading calendar integration (Phase 1 - exchange-calendars)
        self.calendar = None
        if exchange:
            try:
                from core.trading_calendar import TradingCalendar
                self.calendar = TradingCalendar(exchange)
            except Exception as e:
                warnings.warn(f"Could not load trading calendar: {e}")
    
    def run_backtest(self,
                    df: pd.DataFrame,
                    signals: np.ndarray,
                    signal_threshold: float = 0.3,
                    position_size: float = 0.1) -> Dict:
        """
        Run backtest with signals and validation.
        Filters to trading days if exchange calendar is configured.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Array of signals (-1 to 1)
            signal_threshold: Minimum signal strength to act
            position_size: Fraction of capital per position
        
        Returns:
            Backtest results dictionary
        """
        # Validate inputs
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")
        if len(signals) != len(df):
            raise ValueError(f"Signals length ({len(signals)}) must match data length ({len(df)})")
        if position_size <= 0 or position_size > 1:
            raise ValueError("Position size must be between 0 and 1")
        
        # Filter to trading days if calendar is available
        if self.calendar:
            try:
                # Get trading days in the date range
                start_date = df.index[0].strftime('%Y-%m-%d')
                end_date = df.index[-1].strftime('%Y-%m-%d')
                trading_days = self.calendar.trading_days(start_date, end_date)
                
                # Filter DataFrame to only trading days
                df_copy = df.copy()
                df_copy.index = pd.to_datetime(df_copy.index)
                
                # Create a mask for trading days
                mask = df_copy.index.normalize().isin(trading_days.normalize())
                df = df_copy[mask]
                signals = signals[mask.values]
                
                if len(df) == 0:
                    raise ValueError("No trading days found in date range")
            except Exception as e:
                warnings.warn(f"Trading calendar filtering failed: {e}. Proceeding with all days.")
        
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        equity = self.initial_capital
        
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = signals[i]
            
            # Process signal
            position_type = None
            if signal > signal_threshold:
                position_type = 'long'
            elif signal < -signal_threshold:
                position_type = 'short'
            
            # Close opposite positions if signal flips
            if position_type and 'position' in self.open_positions:
                old_position = self.open_positions['position']
                if old_position['type'] != position_type:
                    # Close old position
                    trade = old_position['trade']
                    trade.close(date, price)
                    self.trades.append(trade)
                    
                    # Update equity
                    equity -= trade.pnl
                    equity *= (1 - self.commission)  # Commission on exit
                    
                    self.open_positions = {}
            
            # Open new position
            if position_type and 'position' not in self.open_positions:
                position_qty = (equity * position_size) / price
                trade = Trade(
                    entry_date=date,
                    entry_price=price,
                    quantity=position_qty,
                    position_type=position_type
                )
                self.open_positions['position'] = {
                    'trade': trade,
                    'type': position_type
                }
                equity *= (1 - self.commission)  # Commission on entry
            
            # Mark-to-market equity
            marked_equity = equity
            if 'position' in self.open_positions:
                trade = self.open_positions['position']['trade']
                if trade.position_type == 'long':
                    unrealized_pnl = (price - trade.entry_price) * trade.quantity
                else:
                    unrealized_pnl = (trade.entry_price - price) * trade.quantity
                marked_equity = equity + unrealized_pnl
            
            self.equity_curve.append(marked_equity)
        
        # Close any remaining positions
        if 'position' in self.open_positions:
            trade = self.open_positions['position']['trade']
            trade.close(df.index[-1], df['Close'].iloc[-1])
            self.trades.append(trade)
            equity -= trade.pnl
            equity *= (1 - self.commission)
        
        return self._calculate_metrics(equity)
    
    def _calculate_metrics(self, final_equity: float) -> Dict:
        """Calculate backtest metrics."""
        equity_array = np.array(self.equity_curve)
        
        if len(self.trades) == 0:
            total_return = 0
            win_rate = 0
            avg_pnl = 0
        else:
            total_return = (final_equity - self.initial_capital) / self.initial_capital
            winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
            win_rate = winning_trades / len(self.trades) if self.trades else 0
            avg_pnl = np.mean([t.pnl for t in self.trades])
        
        # Sharpe ratio
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax
        max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': len(self.trades),
            'winning_trades': sum(1 for t in self.trades if t.pnl > 0),
            'losing_trades': sum(1 for t in self.trades if t.pnl < 0),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': sum(t.pnl for t in self.trades),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100
        }


class WalkForwardAnalysis:
    """
    Walk-forward optimization for robust backtesting.
    """
    
    def __init__(self, df: pd.DataFrame, in_sample_period: int = 252, out_sample_period: int = 63):
        """
        Initialize walk-forward analysis.
        
        Args:
            df: Full price data
            in_sample_period: Training period (days)
            out_sample_period: Testing period (days)
        """
        self.df = df
        self.in_sample_period = in_sample_period
        self.out_sample_period = out_sample_period
    
    def run(self, predictor_class, **kwargs) -> List[Dict]:
        """
        Run walk-forward analysis.
        
        Args:
            predictor_class: ML predictor class
            **kwargs: Additional arguments for predictor
        
        Returns:
            List of out-of-sample results
        """
        results = []
        
        # Walk forward windows
        total_length = len(self.df)
        start_idx = 0
        
        while start_idx + self.in_sample_period + self.out_sample_period <= total_length:
            # Training period
            train_end = start_idx + self.in_sample_period
            train_data = self.df.iloc[start_idx:train_end]
            
            # Out-of-sample testing
            test_start = train_end
            test_end = test_start + self.out_sample_period
            test_data = self.df.iloc[test_start:test_end]
            
            # Generate signals
            predictor = predictor_class(**kwargs)
            signals = predictor.predict(test_data)
            
            # Backtest
            engine = BacktestEngine()
            result = engine.run_backtest(test_data, signals)
            result['period'] = (test_data.index[0], test_data.index[-1])
            results.append(result)
            
            # Move forward
            start_idx += self.out_sample_period
        
        return results
