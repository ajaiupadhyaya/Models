"""
Trading Automation
Automated trading with ML signals, risk management, and execution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.backtesting import BacktestEngine
from models.trading.strategies import MomentumStrategy, MeanReversionStrategy
from automation.ml_pipeline import MLPipeline

logger = logging.getLogger(__name__)


class SimplePortfolioTracker:
    """Simple portfolio tracker for automation without broker."""
    
    def __init__(self, initial_capital: float):
        """Initialize portfolio tracker."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', 0)
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trades.copy()
    
    def place_order(self, symbol: str, quantity: int, order_type: str, side: str) -> Dict:
        """Place order (simplified simulation)."""
        # This is a simplified simulation - in production, use actual broker
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'order_type': order_type,
            'status': 'filled',
            'timestamp': datetime.now()
        }
        self.trades.append(order)
        return order


class TradingAutomation:
    """
    Automated trading system with ML signals and risk management.
    """
    
    def __init__(self, 
                 trading_enabled: bool = False,
                 initial_capital: float = 100000.0):
        """
        Initialize trading automation.
        
        Args:
            trading_enabled: Whether to enable live trading
            initial_capital: Initial capital for paper trading
        """
        self.trading_enabled = trading_enabled
        self.initial_capital = initial_capital
        
        # Initialize components
        # Note: PaperTradingEngine requires broker adapter, so we use SimplePortfolioTracker for simulation
        # For actual paper trading, initialize PaperTradingEngine with broker adapter separately
        self.ml_pipeline = MLPipeline()
        self.backtest_engine = BacktestEngine(initial_capital=initial_capital, commission=0.001)
        self.portfolio_tracker = SimplePortfolioTracker(initial_capital)
        
        # Trading state
        self.positions: Dict[str, Dict] = {}
        self.signals: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info(f"Trading automation initialized (enabled: {trading_enabled})")
    
    def generate_ml_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate trading signal using ML models.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Signal dictionary with prediction and confidence
        """
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            
            # Get recent data
            data = fetcher.get_stock_data(symbol, period='3mo')
            
            if len(data) < 50:
                return {
                    'success': False,
                    'error': 'Insufficient data'
                }
            
            # Try to use trained model
            model_id = f"{symbol}_ensemble"
            if model_id in self.ml_pipeline.trained_models:
                model = self.ml_pipeline.trained_models[model_id]['model']
                prediction = model.predict(data.tail(50))
                
                # Convert prediction to signal (-1 to 1)
                if hasattr(prediction, '__len__') and len(prediction) > 0:
                    signal_value = float(prediction[-1])
                else:
                    signal_value = float(prediction)
                
                # Normalize to -1 to 1
                signal_value = np.clip(signal_value, -1, 1)
                
                return {
                    'success': True,
                    'symbol': symbol,
                    'signal': signal_value,
                    'signal_type': 'buy' if signal_value > 0.1 else 'sell' if signal_value < -0.1 else 'hold',
                    'confidence': abs(signal_value),
                    'prediction': float(signal_value),
                    'timestamp': datetime.now()
                }
            else:
                # Use simple momentum strategy as fallback
                strategy = MomentumStrategy(lookback_period=20)
                signals = strategy.generate_signals(data)
                
                if len(signals) > 0:
                    latest_signal = signals.iloc[-1]
                    return {
                        'success': True,
                        'symbol': symbol,
                        'signal': float(latest_signal),
                        'signal_type': 'buy' if latest_signal > 0 else 'sell' if latest_signal < 0 else 'hold',
                        'confidence': abs(float(latest_signal)),
                        'method': 'momentum_fallback',
                        'timestamp': datetime.now()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No signal generated'
                    }
                    
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_trade(self, 
                     symbol: str,
                     signal: Dict[str, Any],
                     position_size: float = 0.1) -> Dict[str, Any]:
        """
        Execute trade based on signal.
        
        Args:
            symbol: Stock symbol
            signal: Signal dictionary
            position_size: Fraction of portfolio to allocate
        
        Returns:
            Trade execution result
        """
        if not self.trading_enabled:
            return {
                'success': False,
                'error': 'Trading not enabled'
            }
        
        try:
            signal_value = signal.get('signal', 0)
            signal_type = signal.get('signal_type', 'hold')
            
            if signal_type == 'hold':
                return {
                    'success': False,
                    'message': 'Hold signal - no trade executed'
                }
            
            # Get current price
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            data = fetcher.get_stock_data(symbol, period='1d')
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate position size
            portfolio_value = self.portfolio_tracker.get_portfolio_value()
            trade_value = portfolio_value * position_size * abs(signal_value)
            shares = int(trade_value / current_price)
            
            if shares == 0:
                return {
                    'success': False,
                    'message': 'Insufficient capital for trade'
                }
            
            # Execute trade (simplified simulation)
            if signal_type == 'buy':
                order = self.portfolio_tracker.place_order(
                    symbol=symbol,
                    quantity=shares,
                    order_type='market',
                    side='buy'
                )
                # Update portfolio
                cost = shares * current_price
                if cost <= self.portfolio_tracker.cash:
                    self.portfolio_tracker.cash -= cost
                    if symbol in self.portfolio_tracker.positions:
                        self.portfolio_tracker.positions[symbol]['quantity'] += shares
                    else:
                        self.portfolio_tracker.positions[symbol] = {
                            'quantity': shares,
                            'current_price': current_price
                        }
            else:  # sell
                # Check if we have position
                positions = self.portfolio_tracker.get_positions()
                if symbol not in positions or positions[symbol].get('quantity', 0) == 0:
                    return {
                        'success': False,
                        'message': f'No position to sell for {symbol}'
                    }
                
                # Sell all or partial
                current_qty = positions[symbol]['quantity']
                sell_qty = min(shares, current_qty)
                
                order = self.portfolio_tracker.place_order(
                    symbol=symbol,
                    quantity=sell_qty,
                    order_type='market',
                    side='sell'
                )
                # Update portfolio
                proceeds = sell_qty * current_price
                self.portfolio_tracker.cash += proceeds
                self.portfolio_tracker.positions[symbol]['quantity'] -= sell_qty
                if self.portfolio_tracker.positions[symbol]['quantity'] == 0:
                    del self.portfolio_tracker.positions[symbol]
            
            return {
                'success': True,
                'symbol': symbol,
                'order': order,
                'signal': signal,
                'shares': shares,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_automated_trading_cycle(self, 
                                    symbols: List[str],
                                    position_size: float = 0.1) -> Dict[str, Any]:
        """
        Run one cycle of automated trading.
        
        Args:
            symbols: List of symbols to trade
            position_size: Position size per trade
        
        Returns:
            Cycle results
        """
        results = {
            'timestamp': datetime.now(),
            'symbols_processed': len(symbols),
            'signals_generated': {},
            'trades_executed': {},
            'errors': []
        }
        
        for symbol in symbols:
            try:
                # Generate signal
                signal = self.generate_ml_signal(symbol)
                results['signals_generated'][symbol] = signal
                
                if signal.get('success') and signal.get('signal_type') != 'hold':
                    # Execute trade
                    trade_result = self.execute_trade(symbol, signal, position_size)
                    results['trades_executed'][symbol] = trade_result
                
            except Exception as e:
                logger.error(f"Error in trading cycle for {symbol}: {e}")
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return results
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            portfolio_value = self.portfolio_tracker.get_portfolio_value()
            positions = self.portfolio_tracker.get_positions()
            trades = self.portfolio_tracker.get_trade_history()
            
            # Calculate returns
            total_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            # Calculate win rate
            if len(trades) > 0:
                profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
                win_rate = profitable_trades / len(trades)
            else:
                win_rate = 0.0
            
            self.performance_metrics = {
                'portfolio_value': portfolio_value,
                'initial_capital': self.initial_capital,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'num_positions': len([p for p in positions.values() if p['quantity'] > 0]),
                'num_trades': len(trades),
                'win_rate': win_rate,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report.
        
        Returns:
            Performance report dictionary
        """
        return {
            'metrics': self.performance_metrics,
            'positions': self.portfolio_tracker.get_positions(),
            'recent_trades': self.portfolio_tracker.get_trade_history()[-10:],
            'timestamp': datetime.now()
        }
    
    def backtest_strategy(self,
                         symbol: str,
                         strategy_type: str = 'momentum',
                         start_date: str = None,
                         end_date: str = None) -> Dict[str, Any]:
        """
        Backtest trading strategy.
        
        Args:
            symbol: Stock symbol
            strategy_type: Strategy type ('momentum', 'mean_reversion')
            start_date: Start date
            end_date: End date
        
        Returns:
            Backtest results
        """
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            
            # Fetch data
            if start_date and end_date:
                data = fetcher.get_stock_data(symbol, start_date=start_date, end_date=end_date)
            else:
                data = fetcher.get_stock_data(symbol, period='1y')
            
            # Select strategy
            if strategy_type == 'momentum':
                strategy = MomentumStrategy(lookback_period=20)
            elif strategy_type == 'mean_reversion':
                strategy = MeanReversionStrategy(lookback_period=20)
            else:
                raise ValueError(f"Unknown strategy: {strategy_type}")
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(data, signals)
            
            return {
                'success': True,
                'symbol': symbol,
                'strategy': strategy_type,
                'results': results,
                'period': {
                    'start': str(data.index[0]),
                    'end': str(data.index[-1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Example usage
    trading = TradingAutomation(trading_enabled=False)  # Paper trading only
    
    # Generate signal
    signal = trading.generate_ml_signal("AAPL")
    print(f"Signal: {signal}")
    
    # Backtest
    backtest = trading.backtest_strategy("AAPL", strategy_type='momentum')
    print(f"Backtest: {backtest}")
