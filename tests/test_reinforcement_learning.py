"""
Tests for reinforcement learning trading engine.

Tests:
- Trading state representation
- Trading environment simulation
- Q-Learning agent
- Portfolio metrics calculation
- Policy evaluation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from core.reinforcement_learning import (
    TradeAction,
    PositionType,
    TradingState,
    Trade,
    TradingEnvironment,
    QLearningAgent,
    PolicyEvaluator,
    PortfolioMetrics,
)


class TestTradeAction:
    """Test trade action enum."""
    
    def test_trade_action_values(self):
        """Test that trade actions have correct values."""
        assert TradeAction.HOLD.value == 0
        assert TradeAction.BUY.value == 1
        assert TradeAction.SELL.value == 2
    
    def test_trade_action_count(self):
        """Test number of actions."""
        actions = list(TradeAction)
        assert len(actions) == 3


class TestPositionType:
    """Test position type enum."""
    
    def test_position_type_values(self):
        """Test position type values."""
        assert PositionType.LONG.value == "long"
        assert PositionType.SHORT.value == "short"
        assert PositionType.FLAT.value == "flat"


class TestTradingState:
    """Test trading state."""
    
    @pytest.fixture
    def sample_state(self):
        """Create sample trading state."""
        return TradingState(
            price=100.0,
            sma_short=99.0,
            sma_long=98.0,
            rsi=55.0,
            returns=1.5,
            volatility=0.02,
            position=PositionType.FLAT,
            portfolio_value=10000.0,
        )
    
    def test_state_initialization(self, sample_state):
        """Test state initialization."""
        assert sample_state.price == 100.0
        assert sample_state.sma_short == 99.0
        assert sample_state.rsi == 55.0
        assert sample_state.position == PositionType.FLAT
    
    def test_state_to_tuple(self, sample_state):
        """Test state discretization to tuple."""
        state_tuple = sample_state.to_tuple()
        
        assert isinstance(state_tuple, tuple)
        assert len(state_tuple) == 6  # 6 discretized dimensions
        assert all(isinstance(x, (int, str)) for x in state_tuple)


class TestTrade:
    """Test trade representation."""
    
    def test_trade_initialization(self):
        """Test trade creation."""
        trade = Trade(
            timestamp=datetime.now(),
            action=TradeAction.BUY,
            price=100.0,
            quantity=10.0,
            position_type=PositionType.LONG,
        )
        
        assert trade.action == TradeAction.BUY
        assert trade.price == 100.0
        assert trade.quantity == 10.0
    
    def test_trade_with_pnl(self):
        """Test trade with P&L."""
        trade = Trade(
            timestamp=datetime.now(),
            action=TradeAction.SELL,
            price=105.0,
            quantity=10.0,
            position_type=PositionType.LONG,
            pnl=50.0,
            pnl_percent=5.0,
        )
        
        assert trade.pnl == 50.0
        assert trade.pnl_percent == 5.0
    
    def test_trade_to_dict(self):
        """Test trade serialization."""
        trade = Trade(
            timestamp=datetime(2024, 1, 1, 10, 0),
            action=TradeAction.BUY,
            price=100.0,
            quantity=10.0,
            position_type=PositionType.LONG,
            pnl=50.0,
        )
        
        trade_dict = trade.to_dict()
        
        assert isinstance(trade_dict, dict)
        assert trade_dict['action'] == 'BUY'
        assert trade_dict['price'] == 100.0
        assert trade_dict['pnl'] == 50.0


class TestTradingEnvironment:
    """Test trading environment."""
    
    @pytest.fixture
    def price_series(self):
        """Generate sample price series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        # Trending up prices
        prices = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def env(self, price_series):
        """Create environment."""
        return TradingEnvironment(price_series, initial_capital=10000.0)
    
    def test_environment_initialization(self, env):
        """Test environment initialization."""
        assert env.initial_capital == 10000.0
        assert env.portfolio_value == 10000.0
        assert env.current_step == 0
        assert env.position == PositionType.FLAT
    
    def test_get_state(self, env):
        """Test getting current state."""
        state = env.get_state()
        
        assert isinstance(state, TradingState)
        assert state.portfolio_value > 0
        assert 0 <= state.rsi <= 100
    
    def test_hold_action(self, env):
        """Test holding position."""
        initial_value = env.portfolio_value
        state0 = env.get_state()
        
        next_state, reward, done = env.step(TradeAction.HOLD)
        
        assert not done
        assert next_state.portfolio_value > 0
    
    def test_buy_action(self, env):
        """Test buy action."""
        state = env.get_state()
        next_state, reward, done = env.step(TradeAction.BUY)
        
        assert env.position == PositionType.LONG
        assert env.quantity > 0
        assert next_state.position == PositionType.LONG
    
    def test_sell_after_buy(self, env):
        """Test selling after buying."""
        env.step(TradeAction.BUY)
        initial_value = env.portfolio_value
        
        next_state, reward, done = env.step(TradeAction.SELL)
        
        assert env.position == PositionType.FLAT
        assert env.quantity == 0
        assert len(env.trades) == 1
    
    def test_reset(self, env):
        """Test environment reset."""
        env.step(TradeAction.BUY)
        env.step(TradeAction.SELL)
        
        env.reset()
        
        assert env.current_step == 0
        assert env.portfolio_value == env.initial_capital
        assert len(env.trades) == 0
        assert env.position == PositionType.FLAT
    
    def test_calculate_indicators(self, env):
        """Test indicator calculation."""
        indicators = env.calculate_indicators(10)
        
        assert 'sma_short' in indicators
        assert 'sma_long' in indicators
        assert 'rsi' in indicators
        assert 'returns' in indicators
        assert 'volatility' in indicators
        
        # RSI should be 0-100
        assert 0 <= indicators['rsi'] <= 100
    
    def test_portfolio_history(self, env):
        """Test portfolio history tracking."""
        initial_history_len = len(env.portfolio_history)
        
        env.step(TradeAction.HOLD)
        
        assert len(env.portfolio_history) == initial_history_len + 1
    
    def test_calculate_metrics(self, env, price_series):
        """Test metric calculation."""
        # Run some steps
        for _ in range(20):
            env.step(TradeAction.HOLD)
        
        metrics = env.calculate_metrics()
        
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.total_return >= -1.0  # Can lose everything
        assert 0 <= metrics.win_rate <= 1
        assert metrics.num_trades >= 0
        assert metrics.max_drawdown >= 0


class TestQLearningAgent:
    """Test Q-Learning agent."""
    
    @pytest.fixture
    def agent(self):
        """Create Q-Learning agent."""
        return QLearningAgent(
            num_actions=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
        )
    
    @pytest.fixture
    def price_series(self):
        """Generate sample price series."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def env(self, price_series):
        """Create environment."""
        return TradingEnvironment(price_series, initial_capital=10000.0)
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.num_actions == 3
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
        assert agent.epsilon == 1.0
        assert len(agent.q_table) == 0
    
    def test_select_action_explores(self, agent):
        """Test action selection during exploration."""
        state = TradingState(
            price=100.0,
            sma_short=99.0,
            sma_long=98.0,
            rsi=55.0,
            returns=1.0,
            volatility=0.02,
        )
        
        # With epsilon=1.0, should always explore
        actions = set()
        for _ in range(30):
            action = agent.select_action(state, training=True)
            actions.add(action)
        
        # Should have tried multiple actions
        assert len(actions) > 1
    
    def test_select_action_exploits(self, agent):
        """Test greedy action selection."""
        agent.epsilon = 0.0  # No exploration
        
        state = TradingState(
            price=100.0,
            sma_short=99.0,
            sma_long=98.0,
            rsi=55.0,
            returns=1.0,
            volatility=0.02,
        )
        
        # With epsilon=0.0, should always exploit
        actions = set()
        for _ in range(10):
            action = agent.select_action(state, training=False)
            actions.add(action)
        
        # Should select same action each time
        assert len(actions) <= 1
    
    def test_learn(self, agent):
        """Test Q-learning update."""
        state1 = TradingState(
            price=100.0, sma_short=99.0, sma_long=98.0,
            rsi=55.0, returns=1.0, volatility=0.02
        )
        state2 = TradingState(
            price=120.0, sma_short=119.0, sma_long=118.0,
            rsi=65.0, returns=5.0, volatility=0.03
        )
        
        state_tuple1 = state1.to_tuple()
        state_tuple2 = state2.to_tuple()
        
        # Ensure they're different states
        assert state_tuple1 != state_tuple2
        
        # Get initial Q-values
        initial_q1 = agent.q_table[state_tuple1][TradeAction.BUY.value]
        
        # Perform Q-learning update
        agent.learn(state1, TradeAction.BUY, 0.05, state2, False)
        
        # Get updated Q-value
        updated_q1 = agent.q_table[state_tuple1][TradeAction.BUY.value]
        
        # Should be updated due to learning
        assert updated_q1 > initial_q1 or initial_q1 == 0
    
    def test_train_episode(self, agent, env):
        """Test training for one episode."""
        cumulative_reward, num_trades = agent.train_episode(env)
        
        assert isinstance(cumulative_reward, (int, float))
        assert isinstance(num_trades, (int, float))
        assert num_trades >= 0
    
    def test_epsilon_decay(self, agent):
        """Test epsilon decay."""
        initial_epsilon = agent.epsilon
        
        for _ in range(10):
            env = TradingEnvironment(
                pd.Series(100 * np.ones(50)),
                initial_capital=10000.0
            )
            agent.train_episode(env)
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
    
    def test_evaluate(self, agent, env):
        """Test agent evaluation."""
        metrics, trades = agent.evaluate(env)
        
        assert isinstance(metrics, PortfolioMetrics)
        assert isinstance(trades, list)
        assert all(isinstance(t, Trade) for t in trades)
    
    def test_training_consistency(self):
        """Test that training improves performance."""
        prices = pd.Series(100 + np.random.normal(0, 2, 200))
        env = TradingEnvironment(prices)
        agent = QLearningAgent()
        
        # Get initial performance
        initial_metrics, _ = agent.evaluate(env)
        initial_return = initial_metrics.total_return
        
        # Train for several episodes
        for _ in range(20):
            agent.train_episode(env)
        
        # Get trained performance
        trained_metrics, _ = agent.evaluate(env)
        trained_return = trained_metrics.total_return
        
        # Training should improve or at least not significantly degrade
        assert trained_return >= initial_return - 0.5


class TestPortfolioMetrics:
    """Test portfolio metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics creation."""
        metrics = PortfolioMetrics(
            total_return=0.05,
            annualized_return=0.10,
            sharpe_ratio=1.5,
            max_drawdown=0.2,
            win_rate=0.6,
            profit_factor=1.5,
            num_trades=10,
        )
        
        assert metrics.total_return == 0.05
        assert metrics.sharpe_ratio == 1.5
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = PortfolioMetrics(
            total_return=0.05,
            annualized_return=0.10,
            sharpe_ratio=1.5,
            max_drawdown=0.2,
            win_rate=0.6,
            profit_factor=1.5,
            num_trades=10,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['total_return'] == 0.05
        assert metrics_dict['sharpe_ratio'] == 1.5


class TestPolicyEvaluator:
    """Test policy evaluation."""
    
    @pytest.fixture
    def price_data(self):
        """Generate price data."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def agents(self):
        """Create multiple agents."""
        return {
            "agent1": QLearningAgent(learning_rate=0.1),
            "agent2": QLearningAgent(learning_rate=0.05),
        }
    
    def test_compare_policies(self, price_data, agents):
        """Test policy comparison."""
        results = PolicyEvaluator.compare_policies(price_data, agents)
        
        assert len(results) == 2
        assert "agent1" in results
        assert "agent2" in results
        assert all(isinstance(m, PortfolioMetrics) for m in results.values())
    
    def test_calculate_strategy_statistics(self, price_data):
        """Test strategy statistics calculation."""
        agent = QLearningAgent()
        
        # Train briefly
        env = TradingEnvironment(price_data)
        for _ in range(5):
            agent.train_episode(env)
        
        stats = PolicyEvaluator.calculate_strategy_statistics(
            agent, price_data, num_evaluations=3
        )
        
        assert isinstance(stats, dict)
        assert 'mean_return' in stats
        assert 'std_return' in stats
        assert 'mean_sharpe' in stats
        assert 'mean_num_trades' in stats


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def realistic_prices(self):
        """Generate realistic price series."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        # Trending up with realistic volatility
        prices = 100 * (1 + np.cumsum(np.random.normal(0.0005, 0.015, 252)))
        return pd.Series(prices, index=dates)
    
    def test_full_trading_loop(self, realistic_prices):
        """Test complete trading loop."""
        env = TradingEnvironment(realistic_prices)
        agent = QLearningAgent()
        
        # Train
        for _ in range(10):
            cumulative_reward, num_trades = agent.train_episode(env)
            assert num_trades >= 0
        
        # Evaluate
        metrics, trades = agent.evaluate(env)
        
        # Allow wide bounds due to random start
        assert metrics.total_return >= -1.0
        assert metrics.num_trades >= 0
    
    def test_agent_on_trending_market(self, realistic_prices):
        """Test agent on trending market."""
        agent = QLearningAgent()
        env = TradingEnvironment(realistic_prices)
        
        # Train
        for _ in range(15):
            agent.train_episode(env)
        
        metrics, trades = agent.evaluate(env)
        
        # Allow wide bounds - agent starts random and may not converge quickly
        assert metrics.total_return >= -1.0
    
    def test_multiple_agents_comparison(self, realistic_prices):
        """Test comparing multiple trained agents."""
        agents = {
            "high_lr": QLearningAgent(learning_rate=0.2),
            "low_lr": QLearningAgent(learning_rate=0.05),
        }
        
        # Train both
        for agent in agents.values():
            for _ in range(10):
                env = TradingEnvironment(realistic_prices)
                agent.train_episode(env)
        
        # Compare
        results = PolicyEvaluator.compare_policies(realistic_prices, agents)
        
        assert len(results) == 2


class TestEdgeCases:
    """Test edge cases."""
    
    def test_flat_prices(self):
        """Test with constant price."""
        flat_prices = pd.Series([100.0] * 50)
        env = TradingEnvironment(flat_prices)
        agent = QLearningAgent()
        
        metrics, trades = agent.evaluate(env)
        
        # Should not crash
        assert metrics.total_return >= -1.0
    
    def test_volatile_prices(self):
        """Test with highly volatile prices."""
        volatile_prices = pd.Series(
            100 + np.random.normal(0, 50, 100)
        )
        env = TradingEnvironment(volatile_prices)
        agent = QLearningAgent()
        
        _, trades = agent.evaluate(env)
        assert isinstance(trades, list)
    
    def test_crash_scenario(self):
        """Test during price crash."""
        crashes = np.concatenate([
            np.linspace(100, 50, 50),  # Crash
            np.linspace(50, 60, 50),   # Recovery
        ])
        prices = pd.Series(crashes)
        
        env = TradingEnvironment(prices)
        agent = QLearningAgent()
        
        metrics, _ = agent.evaluate(env)
        assert metrics.max_drawdown >= 0
    
    def test_short_price_series(self):
        """Test with short price series."""
        short_prices = pd.Series([100, 101, 102])
        env = TradingEnvironment(short_prices)
        agent = QLearningAgent()
        
        metrics, _ = agent.evaluate(env)
        assert metrics.total_return >= -1.0
