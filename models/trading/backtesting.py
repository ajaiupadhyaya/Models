"""
Advanced Backtesting Framework
Institutional-grade backtesting with transaction costs, slippage, and realistic constraints.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    """
    Professional backtesting engine with realistic market simulation.
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005,   # 0.05% slippage
                 min_trade_size: float = 100,
                 max_position_size: float = 0.25):  # 25% max position
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate
            min_trade_size: Minimum trade size
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_trade_size = min_trade_size
        self.max_position_size = max_position_size
        
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def run_backtest(self,
                    prices: pd.DataFrame,
                    signals: pd.Series,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run backtest on strategy.
        
        Args:
            prices: DataFrame with price data (must have 'Close' column)
            signals: Series with trading signals (-1, 0, 1)
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Dictionary with backtest results
        """
        # Filter dates
        if start_date:
            prices = prices[prices.index >= start_date]
            signals = signals[signals.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
            signals = signals[signals.index <= end_date]
        
        # Align indices
        common_index = prices.index.intersection(signals.index)
        prices = prices.loc[common_index]
        signals = signals.loc[common_index]
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Run simulation
        for date in common_index:
            signal = signals.loc[date]
            price = prices.loc[date, 'Close']
            
            # Execute trades
            self._execute_trade(date, signal, price)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(price)
            self.equity_curve.append(portfolio_value)
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                daily_return = (portfolio_value - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_return)
            else:
                self.daily_returns.append(0.0)
        
        # Calculate metrics
        equity_series = pd.Series(self.equity_curve, index=common_index)
        returns_series = pd.Series(self.daily_returns, index=common_index)
        
        metrics = self._calculate_metrics(equity_series, returns_series)
        
        return {
            'equity_curve': equity_series,
            'returns': returns_series,
            'trades': pd.DataFrame(self.trades),
            'metrics': metrics,
            'final_value': equity_series.iloc[-1],
            'total_return': (equity_series.iloc[-1] / self.initial_capital) - 1
        }
    
    def _execute_trade(self, date: datetime, signal: float, price: float):
        """Execute a trade based on signal."""
        current_position = self.positions.get('position', 0)
        target_position = signal
        
        # Calculate desired change
        position_change = target_position - current_position
        
        if abs(position_change) < 0.01:  # No significant change
            return
        
        # Calculate trade size
        portfolio_value = self._calculate_portfolio_value(price)
        max_trade_value = portfolio_value * self.max_position_size
        
        if position_change > 0:  # Buy
            trade_value = min(
                abs(position_change) * portfolio_value,
                max_trade_value,
                self.cash * 0.95  # Leave 5% cash buffer
            )
            
            if trade_value < self.min_trade_size:
                return
            
            # Apply slippage
            execution_price = price * (1 + self.slippage)
            shares = trade_value / execution_price
            
            # Apply commission
            commission_cost = trade_value * self.commission
            total_cost = trade_value + commission_cost
            
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions['position'] = current_position + (trade_value / portfolio_value)
                self.positions['shares'] = self.positions.get('shares', 0) + shares
                
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': execution_price,
                    'shares': shares,
                    'value': trade_value,
                    'commission': commission_cost
                })
        
        elif position_change < 0:  # Sell
            shares_to_sell = abs(position_change) * self.positions.get('shares', 0)
            
            if shares_to_sell < 0.01:
                return
            
            trade_value = shares_to_sell * price
            
            # Apply slippage
            execution_price = price * (1 - self.slippage)
            proceeds = shares_to_sell * execution_price
            
            # Apply commission
            commission_cost = proceeds * self.commission
            net_proceeds = proceeds - commission_cost
            
            self.cash += net_proceeds
            self.positions['shares'] = self.positions.get('shares', 0) - shares_to_sell
            self.positions['position'] = max(0, current_position + position_change)
            
            self.trades.append({
                'date': date,
                'type': 'SELL',
                'price': execution_price,
                'shares': shares_to_sell,
                'value': proceeds,
                'commission': commission_cost
            })
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        shares = self.positions.get('shares', 0)
        position_value = shares * current_price
        return self.cash + position_value
    
    def _calculate_metrics(self,
                          equity_curve: pd.Series,
                          returns: pd.Series) -> Dict:
        """Calculate performance metrics."""
        from ..risk.var_cvar import VaRModel, CVaRModel
        from ...core.utils import calculate_sharpe_ratio, calculate_max_drawdown
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(returns)
        
        # VaR and CVaR
        var_95 = VaRModel.calculate_var(returns, method='historical', confidence_level=0.05)
        cvar_95 = CVaRModel.calculate_cvar(returns, method='historical', confidence_level=0.05)
        
        # Win rate
        trade_returns = []
        if len(self.trades) > 1:
            for i in range(1, len(self.trades)):
                if self.trades[i]['type'] == 'SELL':
                    prev_trade = self.trades[i-1]
                    if prev_trade['type'] == 'BUY':
                        trade_return = (self.trades[i]['price'] - prev_trade['price']) / prev_trade['price']
                        trade_returns.append(trade_return)
        
        win_rate = (np.array(trade_returns) > 0).mean() if len(trade_returns) > 0 else 0
        
        # Profit factor
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [abs(r) for r in trade_returns if r < 0]
        profit_factor = sum(winning_trades) / sum(losing_trades) if sum(losing_trades) > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'num_winning_trades': len(winning_trades) if len(trade_returns) > 0 else 0,
            'num_losing_trades': len(losing_trades) if len(trade_returns) > 0 else 0
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.
    """
    
    def __init__(self, train_period: int = 252, test_period: int = 63):
        """
        Initialize walk-forward optimizer.
        
        Args:
            train_period: Training period length (days)
            test_period: Test period length (days)
        """
        self.train_period = train_period
        self.test_period = test_period
    
    def optimize(self,
                prices: pd.DataFrame,
                strategy_func: Callable,
                param_grid: Dict,
                metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Perform walk-forward optimization.
        
        Args:
            prices: Price data
            strategy_func: Function that generates signals given parameters
            param_grid: Dictionary of parameter ranges
            metric: Metric to optimize
        
        Returns:
            DataFrame with optimization results
        """
        results = []
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        # Walk-forward windows
        total_periods = len(prices)
        windows = []
        
        start = 0
        while start + self.train_period + self.test_period <= total_periods:
            train_end = start + self.train_period
            test_end = train_end + self.test_period
            
            windows.append({
                'train_start': start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })
            
            start += self.test_period  # Rolling window
        
        # Optimize for each window
        for window in windows:
            train_data = prices.iloc[window['train_start']:window['train_end']]
            test_data = prices.iloc[window['test_start']:window['test_end']]
            
            best_params = None
            best_score = -np.inf
            
            # Test all parameter combinations
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                
                # Generate signals
                signals = strategy_func(train_data, **param_dict)
                
                # Backtest on training data
                engine = BacktestEngine()
                result = engine.run_backtest(train_data, signals)
                
                score = result['metrics'].get(metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_params = param_dict
            
            # Test best parameters on test set
            test_signals = strategy_func(test_data, **best_params)
            test_engine = BacktestEngine()
            test_result = test_engine.run_backtest(test_data, test_signals)
            
            results.append({
                'window': len(results),
                'best_params': best_params,
                'train_score': best_score,
                'test_return': test_result['total_return'],
                'test_sharpe': test_result['metrics']['sharpe_ratio']
            })
        
        return pd.DataFrame(results)
