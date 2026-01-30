"""
Fully Automated Trading Orchestrator
Coordinates all AI/ML/DL/RL models for continuous automated trading
"""

import logging
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import json
from pathlib import Path

from core.data_fetcher import DataFetcher
from core.ai_analysis import get_ai_service
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor, RLReadyEnvironment
from models.ml.rl_agents import DQNAgent, PPOAgent, StableBaselines3Wrapper, TradingCallback
from core.paper_trading import AlpacaAdapter
from core.pipeline.data_scheduler import DataScheduler, UpdateFrequency

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    model_source: str  # Which model generated this
    timestamp: datetime


@dataclass
class ModelPerformance:
    """Track model performance metrics."""
    model_name: str
    total_signals: int
    correct_signals: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime


class AutomatedTradingOrchestrator:
    """
    Fully automated trading orchestrator.
    Coordinates ML/DL/RL models, continuous learning, and trade execution.
    """
    
    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
        retrain_frequency: str = "daily",  # daily, weekly, monthly
        use_rl: bool = True,
        use_lstm: bool = True,
        use_ensemble: bool = True,
        risk_limit: float = 0.02  # Max 2% risk per trade
    ):
        """
        Initialize orchestrator.
        
        Args:
            symbols: List of symbols to trade
            initial_capital: Starting capital
            retrain_frequency: How often to retrain models
            use_rl: Enable RL agents
            use_lstm: Enable LSTM models
            use_ensemble: Enable ensemble models
            risk_limit: Maximum risk per trade (as fraction of capital)
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.retrain_frequency = retrain_frequency
        self.use_rl = use_rl
        self.use_lstm = use_lstm
        self.use_ensemble = use_ensemble
        self.risk_limit = risk_limit
        
        # Components
        self.data_fetcher = DataFetcher()
        self.ai_service = get_ai_service()
        self.scheduler = DataScheduler()
        
        # Models
        self.ensemble_models = {}
        self.lstm_models = {}
        self.rl_agents = {}
        self.model_performance = {}
        
        # State
        self.positions = {}
        self.trade_history = []
        self.signals_history = []
        self.last_retrain = {}
        
        # Alpaca adapter (lazy-loaded)
        self._alpaca = None
        
        # Model storage
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Orchestrator initialized for {len(symbols)} symbols")
    
    def get_alpaca_adapter(self) -> Optional[AlpacaAdapter]:
        """Get or initialize Alpaca adapter."""
        if self._alpaca is None:
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_API_SECRET", "")
            base_url = os.getenv("ALPACA_API_BASE", "https://paper-api.alpaca.markets")
            
            if api_key and api_secret:
                try:
                    self._alpaca = AlpacaAdapter(api_key, api_secret, base_url)
                    logger.info("Alpaca adapter initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Alpaca: {e}")
                    self._alpaca = False
            else:
                self._alpaca = False
        
        return self._alpaca if self._alpaca is not False else None
    
    def initialize_models(self):
        """Initialize all models for all symbols."""
        logger.info("Initializing models...")
        
        for symbol in self.symbols:
            try:
                # Fetch historical data
                df = self.data_fetcher.get_stock_data(symbol, period="2y")
                if df is None or len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Initialize ensemble
                if self.use_ensemble:
                    self.ensemble_models[symbol] = EnsemblePredictor(lookback_window=20)
                    self.ensemble_models[symbol].train(df)
                    logger.info(f"Ensemble model trained for {symbol}")
                
                # Initialize LSTM
                if self.use_lstm:
                    self.lstm_models[symbol] = LSTMPredictor(lookback_window=20, hidden_units=64)
                    self.lstm_models[symbol].train(df, epochs=10, verbose=0)
                    logger.info(f"LSTM model trained for {symbol}")
                
                # Initialize RL agent
                if self.use_rl:
                    try:
                        env = RLReadyEnvironment(df, initial_capital=self.initial_capital)
                        state_dim = len(env.reset())
                        action_dim = 4  # hold, long, short, close
                        
                        # Use stable-baselines3 if available, else custom DQN
                        try:
                            agent = StableBaselines3Wrapper(agent_type="PPO")
                            agent.create_agent(env)
                            self.rl_agents[symbol] = agent
                        except:
                            # Fallback to custom DQN
                            agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
                            self.rl_agents[symbol] = agent
                        
                        logger.info(f"RL agent initialized for {symbol}")
                    except Exception as e:
                        logger.warning(f"RL agent failed for {symbol}: {e}")
                
                self.last_retrain[symbol] = datetime.now()
                
            except Exception as e:
                logger.error(f"Failed to initialize models for {symbol}: {e}")
        
        logger.info(f"Initialized models for {len(self.ensemble_models)} symbols")
    
    def generate_signals(self, symbol: str) -> List[TradingSignal]:
        """
        Generate trading signals from all models.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            # Fetch latest data
            df = self.data_fetcher.get_stock_data(symbol, period="3mo")
            if df is None or len(df) < 20:
                return signals
            
            current_price = df['Close'].iloc[-1]
            
            # Ensemble prediction
            if self.use_ensemble and symbol in self.ensemble_models:
                try:
                    ensemble_pred = self.ensemble_models[symbol].predict(df)
                    pred_return = ensemble_pred[-1]
                    
                    if abs(pred_return) > 0.01:  # Significant signal
                        action = "BUY" if pred_return > 0 else "SELL"
                        confidence = min(abs(pred_return) * 10, 1.0)
                        
                        signals.append(TradingSignal(
                            symbol=symbol,
                            action=action,
                            confidence=confidence,
                            price=current_price,
                            target_price=current_price * (1 + pred_return * 2),
                            stop_loss=current_price * (1 - abs(pred_return) * 0.5),
                            reasoning=f"Ensemble model predicts {pred_return*100:.2f}% move",
                            model_source="ensemble",
                            timestamp=datetime.now()
                        ))
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed for {symbol}: {e}")
            
            # LSTM prediction
            if self.use_lstm and symbol in self.lstm_models:
                try:
                    lstm_pred = self.lstm_models[symbol].predict(df)
                    pred_return = lstm_pred[-1]
                    
                    if abs(pred_return) > 0.01:
                        action = "BUY" if pred_return > 0 else "SELL"
                        confidence = min(abs(pred_return) * 10, 1.0)
                        
                        signals.append(TradingSignal(
                            symbol=symbol,
                            action=action,
                            confidence=confidence,
                            price=current_price,
                            target_price=current_price * (1 + pred_return * 2),
                            stop_loss=current_price * (1 - abs(pred_return) * 0.5),
                            reasoning=f"LSTM model predicts {pred_return*100:.2f}% move",
                            model_source="lstm",
                            timestamp=datetime.now()
                        ))
                except Exception as e:
                    logger.warning(f"LSTM prediction failed for {symbol}: {e}")
            
            # RL agent prediction
            if self.use_rl and symbol in self.rl_agents:
                try:
                    env = RLReadyEnvironment(df, initial_capital=self.initial_capital)
                    state = env.reset()
                    
                    agent = self.rl_agents[symbol]
                    if hasattr(agent, 'predict'):
                        action_idx, _ = agent.predict(state, deterministic=True)
                    else:
                        action_idx = agent.act(state, training=False)
                    
                    # Map action to trading signal
                    if action_idx == 1:  # Long
                        signals.append(TradingSignal(
                            symbol=symbol,
                            action="BUY",
                            confidence=0.7,
                            price=current_price,
                            target_price=None,
                            stop_loss=None,
                            reasoning="RL agent recommends long position",
                            model_source="rl",
                            timestamp=datetime.now()
                        ))
                    elif action_idx == 2:  # Short
                        signals.append(TradingSignal(
                            symbol=symbol,
                            action="SELL",
                            confidence=0.7,
                            price=current_price,
                            target_price=None,
                            stop_loss=None,
                            reasoning="RL agent recommends short position",
                            model_source="rl",
                            timestamp=datetime.now()
                        ))
                except Exception as e:
                    logger.warning(f"RL prediction failed for {symbol}: {e}")
            
            # AI analysis
            try:
                if signals:
                    # Get consensus
                    buy_signals = [s for s in signals if s.action == "BUY"]
                    sell_signals = [s for s in signals if s.action == "SELL"]
                    
                    if len(buy_signals) > len(sell_signals):
                        consensus_action = "BUY"
                        avg_confidence = np.mean([s.confidence for s in buy_signals])
                    elif len(sell_signals) > len(buy_signals):
                        consensus_action = "SELL"
                        avg_confidence = np.mean([s.confidence for s in sell_signals])
                    else:
                        consensus_action = "HOLD"
                        avg_confidence = 0.5
                    
                    # Get AI insight
                    insight = self.ai_service.generate_trading_insight(
                        symbol=symbol,
                        current_price=current_price,
                        prediction=current_price * (1 + avg_confidence * 0.02),
                        confidence=avg_confidence,
                        market_context="Automated analysis"
                    )
                    
                    # Add AI-enhanced signal
                    signals.append(TradingSignal(
                        symbol=symbol,
                        action=insight.get("action", consensus_action),
                        confidence=avg_confidence,
                        price=current_price,
                        target_price=None,
                        stop_loss=None,
                        reasoning=insight.get("reasoning", "AI analysis"),
                        model_source="ai",
                        timestamp=datetime.now()
                    ))
            except Exception as e:
                logger.warning(f"AI analysis failed for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
        
        return signals
    
    def execute_trades(self, signals: List[TradingSignal], execute: bool = False) -> List[Dict]:
        """
        Execute trades based on signals.
        
        Args:
            signals: List of trading signals
            execute: Whether to actually execute trades
        
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        if not execute:
            logger.info(f"Simulation mode: {len(signals)} signals generated")
            return executed_trades
        
        alpaca = self.get_alpaca_adapter()
        if not alpaca:
            logger.warning("Alpaca not configured, skipping execution")
            return executed_trades
        
        for signal in signals:
            try:
                # Risk management
                position_size = self.calculate_position_size(signal)
                if position_size <= 0:
                    continue
                
                if signal.action == "BUY":
                    order = alpaca.submit_order(
                        symbol=signal.symbol,
                        qty=position_size,
                        side="buy",
                        type="market"
                    )
                    if order:
                        executed_trades.append({
                            "symbol": signal.symbol,
                            "action": "BUY",
                            "qty": position_size,
                            "price": signal.price,
                            "order_id": order.get("id"),
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.info(f"BUY order executed: {signal.symbol} x{position_size}")
                
                elif signal.action == "SELL":
                    # Check if we have position
                    positions = alpaca.get_positions()
                    if positions and signal.symbol in positions:
                        qty = positions[signal.symbol]["qty"]
                        order = alpaca.submit_order(
                            symbol=signal.symbol,
                            qty=qty,
                            side="sell",
                            type="market"
                        )
                        if order:
                            executed_trades.append({
                                "symbol": signal.symbol,
                                "action": "SELL",
                                "qty": qty,
                                "price": signal.price,
                                "order_id": order.get("id"),
                                "timestamp": datetime.now().isoformat()
                            })
                            logger.info(f"SELL order executed: {signal.symbol} x{qty}")
            
            except Exception as e:
                logger.error(f"Trade execution failed for {signal.symbol}: {e}")
        
        return executed_trades
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """
        Calculate position size based on risk limits.
        
        Args:
            signal: Trading signal
        
        Returns:
            Number of shares
        """
        # Simple position sizing: risk_limit of capital per trade
        risk_amount = self.initial_capital * self.risk_limit
        
        if signal.stop_loss:
            risk_per_share = abs(signal.price - signal.stop_loss)
            if risk_per_share > 0:
                shares = int(risk_amount / risk_per_share)
            else:
                shares = int(risk_amount / (signal.price * 0.02))  # 2% stop loss default
        else:
            shares = int(risk_amount / (signal.price * 0.02))
        
        return max(1, min(shares, 100))  # Between 1 and 100 shares
    
    def retrain_models(self, symbol: Optional[str] = None):
        """
        Retrain models with latest data.
        
        Args:
            symbol: Specific symbol to retrain, or None for all
        """
        symbols_to_retrain = [symbol] if symbol else self.symbols
        
        for sym in symbols_to_retrain:
            try:
                # Check if retraining is needed
                if sym in self.last_retrain:
                    days_since = (datetime.now() - self.last_retrain[sym]).days
                    if self.retrain_frequency == "daily" and days_since < 1:
                        continue
                    elif self.retrain_frequency == "weekly" and days_since < 7:
                        continue
                    elif self.retrain_frequency == "monthly" and days_since < 30:
                        continue
                
                logger.info(f"Retraining models for {sym}...")
                
                # Fetch latest data
                df = self.data_fetcher.get_stock_data(sym, period="2y")
                if df is None or len(df) < 100:
                    continue
                
                # Retrain ensemble
                if self.use_ensemble and sym in self.ensemble_models:
                    self.ensemble_models[sym].train(df)
                
                # Retrain LSTM
                if self.use_lstm and sym in self.lstm_models:
                    self.lstm_models[sym].train(df, epochs=10, verbose=0)
                
                # Retrain RL (if needed)
                if self.use_rl and sym in self.rl_agents:
                    # RL retraining is more expensive, do it less frequently
                    if days_since >= 7:  # Weekly for RL
                        try:
                            env = RLReadyEnvironment(df, initial_capital=self.initial_capital)
                            agent = self.rl_agents[sym]
                            if hasattr(agent, 'train'):
                                callback = TradingCallback()
                                agent.train(total_timesteps=5000, callback=callback)
                        except Exception as e:
                            logger.warning(f"RL retraining failed for {sym}: {e}")
                
                self.last_retrain[sym] = datetime.now()
                logger.info(f"Models retrained for {sym}")
            
            except Exception as e:
                logger.error(f"Retraining failed for {sym}: {e}")
    
    def run_cycle(self, execute_trades: bool = False) -> Dict[str, Any]:
        """
        Run one complete trading cycle.
        
        Args:
            execute_trades: Whether to execute trades
        
        Returns:
            Cycle results
        """
        logger.info("Starting trading cycle...")
        
        all_signals = []
        all_trades = []
        
        # Generate signals for all symbols
        for symbol in self.symbols:
            signals = self.generate_signals(symbol)
            all_signals.extend(signals)
            self.signals_history.extend(signals)
        
        # Execute trades
        if all_signals:
            all_trades = self.execute_trades(all_signals, execute=execute_trades)
            self.trade_history.extend(all_trades)
        
        # Check if retraining is needed
        self.retrain_models()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "signals_generated": len(all_signals),
            "trades_executed": len(all_trades),
            "signals": [asdict(s) for s in all_signals],
            "trades": all_trades
        }
    
    def start_automated_trading(self, interval_minutes: int = 60, execute: bool = False):
        """
        Start continuous automated trading.
        
        Args:
            interval_minutes: Minutes between cycles
            execute: Whether to execute trades
        """
        logger.info(f"Starting automated trading (interval: {interval_minutes} min, execute: {execute})")
        
        # Initialize models
        self.initialize_models()
        
        # Schedule periodic cycles
        def run_cycle_job():
            try:
                result = self.run_cycle(execute_trades=execute)
                logger.info(f"Cycle completed: {result['signals_generated']} signals, {result['trades_executed']} trades")
            except Exception as e:
                logger.error(f"Cycle failed: {e}")
        
        # Add to scheduler
        from core.pipeline.data_scheduler import UpdateJob, UpdateFrequency
        
        job = UpdateJob(
            job_id="automated_trading",
            name="Automated Trading Cycle",
            function=run_cycle_job,
            frequency=UpdateFrequency.INTRADAY
        )
        
        self.scheduler.add_job(job)
        self.scheduler.start()
        
        logger.info("Automated trading started")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "symbols": self.symbols,
            "models_initialized": {
                "ensemble": len(self.ensemble_models),
                "lstm": len(self.lstm_models),
                "rl": len(self.rl_agents)
            },
            "total_signals": len(self.signals_history),
            "total_trades": len(self.trade_history),
            "last_retrain": {k: v.isoformat() for k, v in self.last_retrain.items()},
            "positions": self.positions
        }
