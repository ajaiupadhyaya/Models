"""
Reinforcement Learning for algorithmic trading.

Implements:
- Q-Learning agent for trading decisions
- Trading environment simulation
- Portfolio management and performance tracking
- Reward calculation and policy evaluation
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict


class TradeAction(Enum):
    """Trading actions available to the agent."""
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TradingState:
    """Represents the state of the trading environment."""
    price: float
    sma_short: float  # 10-period SMA
    sma_long: float   # 30-period SMA
    rsi: float        # Relative Strength Index (0-100)
    returns: float    # Recent returns (%)
    volatility: float # Recent volatility
    position: PositionType = PositionType.FLAT
    portfolio_value: float = 10000.0
    timestamp: Optional[datetime] = None
    
    def to_tuple(self) -> Tuple:
        """Convert state to discrete tuple for Q-table indexing."""
        # Discretize continuous state variables
        price_bin = min(10, int(self.price / 10))
        sma_diff = self.sma_short - self.sma_long
        sma_bin = min(10, max(0, int((sma_diff + 50) / 10)))  # -50 to 50 range
        rsi_bin = min(10, int(self.rsi / 10))
        returns_bin = min(10, max(0, int((self.returns + 10) / 2)))  # -10 to 10 range
        vol_bin = min(10, int(self.volatility * 100))
        pos_bin = self.position.value if isinstance(self.position, str) else self.position.name
        
        return (price_bin, sma_bin, rsi_bin, returns_bin, vol_bin, pos_bin)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    action: TradeAction
    price: float
    quantity: float
    position_type: PositionType
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'action': self.action.name,
            'price': self.price,
            'quantity': self.quantity,
            'position_type': self.position_type.value,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
        }


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'num_trades': self.num_trades,
        }


class TradingEnvironment:
    """Simulates a trading environment with price data."""
    
    def __init__(self, price_data: pd.Series, initial_capital: float = 10000.0):
        """
        Initialize trading environment.
        
        Args:
            price_data: Time series of prices
            initial_capital: Starting capital
        """
        self.price_data = price_data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.trades: List[Trade] = []
        self.position = PositionType.FLAT
        self.entry_price = 0.0
        self.quantity = 0.0
        self.portfolio_history: List[float] = [initial_capital]
        
    def calculate_indicators(self, step: int) -> Dict[str, float]:
        """Calculate technical indicators for given step."""
        if step < 30:
            # Not enough data for all indicators
            return {
                'sma_short': self.price_data.iloc[step],
                'sma_long': self.price_data.iloc[step],
                'rsi': 50.0,
                'returns': 0.0,
                'volatility': 0.01,
            }
        
        prices = self.price_data.iloc[:step+1].values
        
        # Simple Moving Averages
        sma_short = np.mean(prices[-10:])  # 10-period
        sma_long = np.mean(prices[-30:])   # 30-period
        
        # RSI: Relative Strength Index
        deltas = np.diff(prices[-14:])
        gains = np.sum(np.maximum(deltas, 0))
        losses = np.sum(np.abs(np.minimum(deltas, 0)))
        
        if losses == 0:
            rsi = 100.0 if gains > 0 else 50.0
        else:
            rs = gains / losses
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Recent returns
        returns = ((prices[-1] - prices[-10]) / prices[-10] * 100) if len(prices) >= 10 else 0.0
        
        # Volatility
        volatility = np.std(np.diff(prices[-20:])) / np.mean(prices[-20:]) if len(prices) >= 20 else 0.01
        
        return {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'rsi': rsi,
            'returns': returns,
            'volatility': volatility,
        }
    
    def get_state(self) -> TradingState:
        """Get current state of the environment."""
        indicators = self.calculate_indicators(self.current_step)
        
        return TradingState(
            price=self.price_data.iloc[self.current_step],
            sma_short=indicators['sma_short'],
            sma_long=indicators['sma_long'],
            rsi=indicators['rsi'],
            returns=indicators['returns'],
            volatility=indicators['volatility'],
            position=self.position,
            portfolio_value=self.portfolio_value,
            timestamp=pd.Timestamp(self.current_step),
        )
    
    def step(self, action: TradeAction) -> Tuple[TradingState, float, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading action to execute
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        current_price = self.price_data.iloc[self.current_step]
        reward = 0.0
        
        if action == TradeAction.BUY and self.position == PositionType.FLAT:
            # Initiate long position
            self.quantity = self.portfolio_value * 0.95 / current_price
            self.entry_price = current_price
            self.position = PositionType.LONG
            self.portfolio_value *= 0.99  # Transaction cost
            
        elif action == TradeAction.SELL and self.position == PositionType.LONG:
            # Close long position
            pnl = (current_price - self.entry_price) * self.quantity
            pnl_percent = (current_price - self.entry_price) / self.entry_price * 100
            
            self.portfolio_value += pnl
            self.portfolio_value *= 0.99  # Transaction cost
            
            trade = Trade(
                timestamp=pd.Timestamp(self.current_step),
                action=action,
                price=current_price,
                quantity=self.quantity,
                position_type=self.position,
                pnl=pnl,
                pnl_percent=pnl_percent,
            )
            self.trades.append(trade)
            
            reward = max(-0.1, min(0.1, pnl_percent / 100))  # Normalize reward
            
            self.position = PositionType.FLAT
            self.quantity = 0.0
        
        # Transition to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.price_data) - 1
        
        if self.position == PositionType.LONG and done:
            # Close position at end
            current_price = self.price_data.iloc[-1]
            pnl = (current_price - self.entry_price) * self.quantity
            self.portfolio_value += pnl
            self.portfolio_value *= 0.99
            self.position = PositionType.FLAT
        
        self.portfolio_history.append(self.portfolio_value)
        
        # Reward based on portfolio performance
        if not done:
            portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            reward += portfolio_return * 0.01  # Small reward for portfolio growth
        
        next_state = self.get_state()
        
        return next_state, reward, done
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.position = PositionType.FLAT
        self.entry_price = 0.0
        self.quantity = 0.0
        self.portfolio_history = [self.initial_capital]
    
    def calculate_metrics(self) -> PortfolioMetrics:
        """Calculate performance metrics."""
        portfolio_array = np.array(self.portfolio_history)
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming 252 trading days)
        num_days = len(self.portfolio_history)
        annualized_return = total_return * (252 / max(1, num_days))
        
        # Sharpe ratio
        sharpe_ratio = 0.0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        max_drawdown = 0.0
        cummax = np.maximum.accumulate(portfolio_array)
        drawdowns = (cummax - portfolio_array) / cummax
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win rate
        win_rate = 0.0
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t.pnl_percent and t.pnl_percent > 0)
            win_rate = wins / len(self.trades)
        
        # Profit factor
        profit_factor = 0.0
        if len(self.trades) > 0:
            gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
            gross_loss = sum(abs(t.pnl) for t in self.trades if t.pnl and t.pnl < 0)
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = float('inf')
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
        )


class QLearningAgent:
    """Q-Learning agent for trading."""
    
    def __init__(
        self,
        num_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            num_actions: Number of possible actions (3: HOLD, BUY, SELL)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary mapping (state) -> [Q-values for each action]
        self.q_table: Dict[Tuple, List[float]] = defaultdict(
            lambda: [0.0] * num_actions
        )
        
        self.training_episode_count = 0
        
    def select_action(self, state: TradingState, training: bool = True) -> TradeAction:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses exploration)
            
        Returns:
            Selected action
        """
        state_tuple = state.to_tuple()
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.num_actions)
        else:
            # Exploit: greedy action
            q_values = self.q_table[state_tuple]
            action_idx = np.argmax(q_values)
        
        return TradeAction(action_idx)
    
    def learn(self, state: TradingState, action: TradeAction, reward: float, next_state: TradingState, done: bool):
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_tuple = state.to_tuple()
        next_state_tuple = next_state.to_tuple()
        action_idx = action.value
        
        # Q-learning update
        current_q = self.q_table[state_tuple][action_idx]
        next_max_q = max(self.q_table[next_state_tuple]) if not done else 0.0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state_tuple][action_idx] = new_q
    
    def train_episode(self, environment: TradingEnvironment) -> Tuple[float, int]:
        """
        Train for one episode.
        
        Args:
            environment: Trading environment
            
        Returns:
            Tuple of (cumulative_reward, num_trades)
        """
        environment.reset()
        state = environment.get_state()
        cumulative_reward = 0.0
        done = False
        
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, done = environment.step(action)
            
            self.learn(state, action, reward, next_state, done)
            cumulative_reward += reward
            state = next_state
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_episode_count += 1
        
        return cumulative_reward, len(environment.trades)
    
    def evaluate(self, environment: TradingEnvironment) -> Tuple[PortfolioMetrics, List[Trade]]:
        """
        Evaluate agent on environment (no learning).
        
        Args:
            environment: Trading environment
            
        Returns:
            Tuple of (metrics, trades)
        """
        environment.reset()
        state = environment.get_state()
        done = False
        
        while not done:
            action = self.select_action(state, training=False)
            next_state, _, done = environment.step(action)
            state = next_state
        
        metrics = environment.calculate_metrics()
        
        return metrics, environment.trades


class PolicyEvaluator:
    """Evaluates different trading policies."""
    
    @staticmethod
    def compare_policies(
        price_data: pd.Series,
        policies: Dict[str, QLearningAgent],
        initial_capital: float = 10000.0,
    ) -> Dict[str, PortfolioMetrics]:
        """
        Compare multiple trading policies.
        
        Args:
            price_data: Price series
            policies: Dictionary of policy_name -> agent
            initial_capital: Starting capital
            
        Returns:
            Dictionary of policy_name -> metrics
        """
        results = {}
        
        for policy_name, agent in policies.items():
            env = TradingEnvironment(price_data, initial_capital)
            metrics, _ = agent.evaluate(env)
            results[policy_name] = metrics
        
        return results
    
    @staticmethod
    def calculate_strategy_statistics(
        agent: QLearningAgent,
        price_data: pd.Series,
        num_evaluations: int = 10,
        initial_capital: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Calculate statistics across multiple strategy evaluations.
        
        Args:
            agent: Trained agent
            price_data: Price series
            num_evaluations: Number of evaluation runs
            initial_capital: Starting capital
            
        Returns:
            Dictionary of statistics
        """
        returns = []
        sharpes = []
        max_dd_list = []
        num_trades_list = []
        
        for _ in range(num_evaluations):
            env = TradingEnvironment(price_data, initial_capital)
            metrics, _ = agent.evaluate(env)
            
            returns.append(metrics.total_return)
            sharpes.append(metrics.sharpe_ratio)
            max_dd_list.append(metrics.max_drawdown)
            num_trades_list.append(metrics.num_trades)
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpes),
            'mean_max_dd': np.mean(max_dd_list),
            'mean_num_trades': np.mean(num_trades_list),
        }
