"""
Advanced Reinforcement Learning Agents for Trading
Implements PPO, DQN, A3C, and other state-of-the-art RL algorithms
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    BaseCallback = None  # type: ignore[misc, assignment]

from .advanced_trading import RLReadyEnvironment


if HAS_SB3:
    class TradingCallback(BaseCallback):
        """Callback for tracking training progress."""
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
        def _on_step(self) -> bool:
            return True
else:
    TradingCallback = None  # type: ignore[misc, assignment]


class DQNAgent:
    """
    Deep Q-Network agent for trading.
    Uses experience replay and target networks for stable learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 64
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            memory_size: Replay buffer size
            batch_size: Batch size for training
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for DQN. Install: pip install torch")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        self.training_step = 0
    
    def _build_network(self) -> nn.Module:
        """Build Q-network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Action index
        """
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self) -> Optional[float]:
        """
        Train on a batch of experiences.
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % 100 == 0:
            self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.training_step = checkpoint.get('training_step', 0)


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    More stable than DQN, better for continuous control.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for PPO. Install: pip install torch")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Actor-Critic network
        self.actor_critic = self._build_actor_critic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def _build_actor_critic(self) -> nn.Module:
        """Build actor-critic network."""
        class ActorCritic(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
                self.actor = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim),
                    nn.Softmax(dim=-1)
                )
                self.critic = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            
            def forward(self, state):
                shared = self.shared(state)
                action_probs = self.actor(shared)
                value = self.critic(shared)
                return action_probs, value
        
        return ActorCritic(self.state_dim, self.action_dim)
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
        
        Returns:
            action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.actor_critic(state_tensor)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            advantages, returns
        """
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, states: List[np.ndarray], actions: List[int], 
              old_log_probs: List[float], advantages: List[float], 
              returns: List[float], epochs: int = 4):
        """
        Update policy using PPO algorithm.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Old log probabilities
            advantages: Computed advantages
            returns: Computed returns
            epochs: Number of update epochs
        """
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        for _ in range(epochs):
            action_probs, values = self.actor_critic(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            # PPO clip
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class StableBaselines3Wrapper:
    """
    Wrapper for stable-baselines3 agents.
    Provides easy integration with RLReadyEnvironment.
    """
    
    def __init__(self, agent_type: str = "PPO", **kwargs):
        """
        Initialize stable-baselines3 agent.
        
        Args:
            agent_type: "PPO", "DQN", or "A2C"
            **kwargs: Agent-specific parameters
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required. Install: pip install stable-baselines3")
        
        self.agent_type = agent_type
        self.agent = None
        self.kwargs = kwargs
    
    def create_agent(self, env):
        """Create agent for environment."""
        if self.agent_type == "PPO":
            self.agent = PPO("MlpPolicy", env, verbose=1, **self.kwargs)
        elif self.agent_type == "DQN":
            self.agent = DQN("MlpPolicy", env, verbose=1, **self.kwargs)
        elif self.agent_type == "A2C":
            self.agent = A2C("MlpPolicy", env, verbose=1, **self.kwargs)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def train(self, total_timesteps: int = 10000, callback: Optional[Any] = None):
        """Train the agent."""
        if self.agent is None:
            raise ValueError("Agent not created. Call create_agent first.")
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        """Predict action."""
        if self.agent is None:
            raise ValueError("Agent not created. Call create_agent first.")
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, filepath: str):
        """Save model."""
        if self.agent is None:
            raise ValueError("Agent not created.")
        self.agent.save(filepath)
    
    def load(self, filepath: str):
        """Load model."""
        if self.agent_type == "PPO":
            self.agent = PPO.load(filepath)
        elif self.agent_type == "DQN":
            self.agent = DQN.load(filepath)
        elif self.agent_type == "A2C":
            self.agent = A2C.load(filepath)
