"""
Institutional-Grade Backtesting Engine
Proper transaction costs, market impact, slippage, and statistical validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from core.backtesting import BacktestEngine, Trade
from models.quant.institutional_grade import TransactionCostModel, AdvancedRiskMetrics, StatisticalValidation


@dataclass
class InstitutionalTrade(Trade):
    """Enhanced trade with transaction costs."""
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    market_impact_entry: float = 0.0
    market_impact_exit: float = 0.0
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    total_cost: float = 0.0


class InstitutionalBacktestEngine(BacktestEngine):
    """
    Institutional-grade backtesting with proper transaction cost modeling.
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 market_impact_alpha: float = 0.5):
        """
        Initialize institutional backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (fraction)
            slippage: Slippage rate (fraction)
            market_impact_alpha: Market impact exponent
        """
        super().__init__(initial_capital, commission)
        self.slippage = slippage
        self.market_impact_alpha = market_impact_alpha
        self.transaction_cost_model = TransactionCostModel()
    
    def calculate_transaction_costs(self,
                                   trade_size: float,
                                   price: float,
                                   daily_volume: float,
                                   volatility: float,
                                   bid_ask_spread: float = 0.001) -> Dict[str, float]:
        """
        Calculate all transaction costs.
        
        Args:
            trade_size: Trade size (shares)
            price: Trade price
            daily_volume: Daily volume
            volatility: Asset volatility
            bid_ask_spread: Bid-ask spread
        
        Returns:
            Dictionary with cost breakdown
        """
        # Market impact
        market_impact = self.transaction_cost_model.calculate_market_impact(
            trade_size, daily_volume, volatility, self.market_impact_alpha
        )
        
        # Slippage (random component)
        slippage_cost = np.random.normal(0, self.slippage)
        
        # Commission
        commission_cost = self.commission
        
        # Bid-ask spread
        spread_cost = bid_ask_spread / 2  # Half spread (entry or exit)
        
        # Total cost
        total_cost_pct = market_impact + abs(slippage_cost) + commission_cost + spread_cost
        
        return {
            'market_impact': float(market_impact),
            'slippage': float(abs(slippage_cost)),
            'commission': float(commission_cost),
            'spread': float(spread_cost),
            'total_cost_pct': float(total_cost_pct),
            'total_cost_dollars': float(trade_size * price * total_cost_pct)
        }
    
    def run_backtest(self,
                    df: pd.DataFrame,
                    signals: np.ndarray,
                    signal_threshold: float = 0.3,
                    position_size: float = 0.1) -> Dict:
        """
        Run institutional-grade backtest with proper transaction costs.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Array of signals
            signal_threshold: Minimum signal strength
            position_size: Fraction of capital per position
        
        Returns:
            Enhanced backtest results
        """
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        equity = self.initial_capital
        
        # Calculate daily volume and volatility for transaction costs
        daily_volume = df['Volume'].rolling(20).mean()
        volatility = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            volume = daily_volume.iloc[i] if not pd.isna(daily_volume.iloc[i]) else df['Volume'].iloc[i]
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.2
            
            signal = signals[i]
            
            # Process signal
            position_type = None
            if signal > signal_threshold:
                position_type = 'long'
            elif signal < -signal_threshold:
                position_type = 'short'
            
            # Close opposite positions
            if position_type and 'position' in self.open_positions:
                old_position = self.open_positions['position']
                if old_position['type'] != position_type:
                    trade = old_position['trade']
                    
                    # Calculate exit costs
                    exit_costs = self.calculate_transaction_costs(
                        trade.quantity, price, volume, vol
                    )
                    
                    trade.close(date, price)
                    trade.exit_cost = exit_costs['total_cost_dollars']
                    trade.total_cost = trade.entry_cost + trade.exit_cost
                    
                    # Adjust PnL for costs
                    trade.pnl -= trade.total_cost
                    
                    self.trades.append(trade)
                    equity += trade.pnl
                    self.open_positions = {}
            
            # Open new position
            if position_type and 'position' not in self.open_positions:
                position_qty = (equity * position_size) / price
                
                # Calculate entry costs
                entry_costs = self.calculate_transaction_costs(
                    position_qty, price, volume, vol
                )
                
                trade = InstitutionalTrade(
                    entry_date=date,
                    entry_price=price,
                    quantity=position_qty,
                    position_type=position_type,
                    entry_cost=entry_costs['total_cost_dollars'],
                    market_impact_entry=entry_costs['market_impact'] * position_qty * price,
                    slippage_entry=entry_costs['slippage'] * position_qty * price
                )
                
                # Adjust entry price for costs
                effective_entry_price = price * (1 + entry_costs['total_cost_pct'])
                trade.entry_price = effective_entry_price
                
                self.open_positions['position'] = {
                    'trade': trade,
                    'type': position_type
                }
                
                equity -= entry_costs['total_cost_dollars']
            
            # Mark-to-market
            marked_equity = equity
            if 'position' in self.open_positions:
                trade = self.open_positions['position']['trade']
                if trade.position_type == 'long':
                    unrealized_pnl = (price - trade.entry_price) * trade.quantity
                else:
                    unrealized_pnl = (trade.entry_price - price) * trade.quantity
                marked_equity = equity + unrealized_pnl
            
            self.equity_curve.append(marked_equity)
        
        # Close remaining positions
        if 'position' in self.open_positions:
            trade = self.open_positions['position']['trade']
            exit_costs = self.calculate_transaction_costs(
                trade.quantity, df['Close'].iloc[-1],
                daily_volume.iloc[-1] if not pd.isna(daily_volume.iloc[-1]) else df['Volume'].iloc[-1],
                volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.2
            )
            trade.close(df.index[-1], df['Close'].iloc[-1])
            trade.exit_cost = exit_costs['total_cost_dollars']
            trade.total_cost = trade.entry_cost + trade.exit_cost
            trade.pnl -= trade.total_cost
            self.trades.append(trade)
            equity += trade.pnl
        
        # Calculate enhanced metrics
        return self._calculate_institutional_metrics(equity, df)
    
    def _calculate_institutional_metrics(self, final_equity: float, df: pd.DataFrame) -> Dict:
        """Calculate institutional-grade metrics."""
        base_metrics = self._calculate_metrics(final_equity)
        equity_series = pd.Series(self.equity_curve, index=df.index[:len(self.equity_curve)])
        returns = equity_series.pct_change().dropna()
        
        # Advanced risk metrics
        max_dd = AdvancedRiskMetrics.maximum_drawdown(equity_series)
        sortino = AdvancedRiskMetrics.sortino_ratio(returns)
        calmar = AdvancedRiskMetrics.calmar_ratio(returns, equity_series)
        tail_ratio = AdvancedRiskMetrics.tail_ratio(returns)
        
        # Expected shortfall
        es_95 = AdvancedRiskMetrics.expected_shortfall(returns, 0.05)
        
        # Transaction cost analysis
        total_costs = sum(t.total_cost for t in self.trades if hasattr(t, 'total_cost'))
        avg_cost_per_trade = total_costs / len(self.trades) if self.trades else 0
        
        # Statistical validation
        validation = StatisticalValidation.normality_test(returns)
        
        enhanced_metrics = {
            **base_metrics,
            'max_drawdown_details': max_dd,
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'tail_ratio': float(tail_ratio),
            'expected_shortfall_95': float(es_95),
            'total_transaction_costs': float(total_costs),
            'avg_cost_per_trade': float(avg_cost_per_trade),
            'cost_as_pct_of_capital': float(total_costs / self.initial_capital * 100),
            'normality_test': validation
        }
        
        return enhanced_metrics
