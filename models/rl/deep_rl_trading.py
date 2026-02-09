"""
Deep Reinforcement Learning for Trading Strategies
Phase 3 - Awesome Quant Integration

Uses stable-baselines3 for RL agents with custom trading environments.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from stable_baselines3 import PPO, A2C, DQN, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
    BaseCallbackType = BaseCallback
except ImportError:
    SB3_AVAILABLE = False
    BaseCallbackType = None  # Use None as placeholder when not available
    warnings.warn("stable-baselines3 not available. Install with: pip install stable-baselines3")


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for reinforcement learning trading.
    
    State: [position, cash, portfolio_value, price_features, technical_indicators]
    Action: [-1, 0, 1] representing sell, hold, buy
    Reward: Portfolio return + risk-adjusted metrics
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        lookback_window: int = 20
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Starting cash
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            reward_scaling: Scale rewards
            lookback_window: Number of previous bars to include in state
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.lookback_window = lookback_window
        
        # State space: [position, cash_ratio, portfolio_value_change, price_features(lookback*4), indicators(5)]
        state_size = 3 + (lookback_window * 4) + 5  # 3 + 80 + 5 = 88
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Action space: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)
        
        # Episode tracking
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.cash = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.trades = []
        
        # Performance tracking
        self.portfolio_values = []
        self.sharpe_ratio = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.position = 0
        self.cash = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        State includes:
        - Position and cash ratio
        - Recent price changes (normalized)
        - Technical indicators
        """
        # Position and cash
        position_state = np.array([
            self.position,
            self.cash / self.initial_balance,
            self.portfolio_value / self.initial_balance - 1
        ])
        
        # Price features (OHLC normalized by close)
        window = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        close_prices = window['Close'].values
        
        price_features = []
        for i in range(len(window)):
            close = close_prices[i]
            norm_open = (window['Open'].iloc[i] - close) / close
            norm_high = (window['High'].iloc[i] - close) / close
            norm_low = (window['Low'].iloc[i] - close) / close
            norm_close = 0.0  # Close is reference
            price_features.extend([norm_open, norm_high, norm_low, norm_close])
        
        price_features = np.array(price_features)
        
        # Technical indicators at current step
        current_close = self.df['Close'].iloc[self.current_step]
        sma_20 = self.df['Close'].iloc[max(0, self.current_step - 20):self.current_step].mean()
        sma_50 = self.df['Close'].iloc[max(0, self.current_step - 50):self.current_step].mean()
        
        returns = self.df['Close'].pct_change()
        momentum = returns.iloc[max(0, self.current_step - 10):self.current_step].sum()
        volatility = returns.iloc[max(0, self.current_step - 20):self.current_step].std()
        
        volume_ratio = self.df['Volume'].iloc[self.current_step] / self.df['Volume'].iloc[max(0, self.current_step - 20):self.current_step].mean()
        
        indicators = np.array([
            (current_close - sma_20) / sma_20 if sma_20 > 0 else 0,
            (current_close - sma_50) / sma_50 if sma_50 > 0 else 0,
            momentum,
            volatility,
            volume_ratio - 1
        ])
        
        # Combine all features
        observation = np.concatenate([position_state, price_features, indicators])
        return observation.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=sell, 1=hold, 2=buy
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Current price
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Execute action
        reward = 0
        action_taken = "hold"
        
        if action == 0 and self.position >= 0:  # Sell (or short if already flat)
            # Sell shares
            if self.shares > 0:
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.shares = 0
                self.position = 0
                action_taken = "sell"
        
        elif action == 2 and self.position <= 0:  # Buy
            # Buy shares with available cash
            shares_to_buy = self.cash / (current_price * (1 + self.transaction_cost))
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)
            
            if cost <= self.cash:
                self.shares += shares_to_buy
                self.cash -= cost
                self.position = 1 if self.shares > 0 else 0
                action_taken = "buy"
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        new_price = self.df['Close'].iloc[self.current_step]
        self.portfolio_value = self.cash + self.shares * new_price
        
        # Calculate reward (portfolio return)
        prev_value = self.portfolio_values[-1]
        portfolio_return = (self.portfolio_value - prev_value) / prev_value
        reward = portfolio_return * self.reward_scaling
        
        # Risk-adjusted reward: penalize volatility
        if len(self.portfolio_values) > 10:
            recent_returns = np.diff(self.portfolio_values[-10:]) / self.portfolio_values[-10:-1]
            volatility = np.std(recent_returns)
            reward -= volatility * 0.1  # Volatility penalty
        
        self.portfolio_values.append(self.portfolio_value)
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Get new observation
        observation = self._get_observation() if not terminated else self._get_observation()
        
        # Info
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "action_taken": action_taken,
            "price": new_price
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment state."""
        print(f"Step: {self.current_step}, Portfolio: ${self.portfolio_value:.2f}, Position: {self.position}")


class RLTrader:
    """
    Reinforcement Learning trader using stable-baselines3.
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        verbose: int = 1
    ):
        """
        Initialize RL trader.
        
        Args:
            algorithm: "PPO", "A2C", "DQN", or "SAC"
            policy: Policy network type
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            verbose: Verbosity level
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 required. Install: pip install stable-baselines3")
        
        self.algorithm = algorithm
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.env = None
    
    def create_environment(self, df: pd.DataFrame, **env_kwargs) -> TradingEnvironment:
        """Create trading environment."""
        self.env = TradingEnvironment(df, **env_kwargs)
        return self.env
    
    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[Any] = None
    ) -> None:
        """
        Train RL agent.
        
        Args:
            total_timesteps: Total training steps
            callback: Optional callback for monitoring (BaseCallback when SB3 available)
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment() first.")
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize model
        if self.algorithm == "PPO":
            self.model = PPO(
                self.policy,
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                verbose=self.verbose
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                self.policy,
                vec_env,
                learning_rate=self.learning_rate,
                verbose=self.verbose
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                self.policy,
                vec_env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Train
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action for given observation.
        
        Args:
            observation: Current state observation
            deterministic: Use deterministic policy
        
        Returns:
            Action (0=sell, 1=hold, 2=buy)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Backtest trained agent on new data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create test environment
        test_env = TradingEnvironment(df)
        obs, _ = test_env.reset()
        
        portfolio_values = [test_env.initial_balance]
        actions = []
        done = False
        
        while not done:
            action = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info['portfolio_value'])
            actions.append(info['action_taken'])
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = np.min(np.minimum.accumulate(portfolio_values) / portfolio_values - 1)
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        return {
            "final_portfolio_value": portfolio_values[-1],
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown * 100,
            "num_trades": len(actions),
            "portfolio_values": portfolio_values,
            "actions": actions
        }
    
    def save(self, path: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
    
    def load(self, path: str, env: TradingEnvironment) -> None:
        """Load trained model."""
        vec_env = DummyVecEnv([lambda: env])
        
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=vec_env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=vec_env)
        elif self.algorithm == "DQN":
            self.model = DQN.load(path, env=vec_env)
