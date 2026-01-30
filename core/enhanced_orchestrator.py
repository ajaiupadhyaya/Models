"""
Enhanced Orchestrator with Advanced Quant Features
Integrates factor models, regime detection, and portfolio optimization
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import pandas as pd
import numpy as np

from core.automated_trading_orchestrator import AutomatedTradingOrchestrator, TradingSignal
from models.quant.advanced_models import FactorModel, RegimeDetector, PortfolioOptimizerAdvanced
from core.performance_optimizer import cached, SmartCache
from typing import Tuple

logger = logging.getLogger(__name__)


class EnhancedOrchestrator(AutomatedTradingOrchestrator):
    """
    Enhanced orchestrator with advanced quantitative features.
    Extends base orchestrator with factor models, regime detection, and portfolio optimization.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced orchestrator."""
        super().__init__(*args, **kwargs)
        
        # Advanced models
        self.factor_model = FactorModel(n_factors=3)
        self.regime_detector = RegimeDetector(n_regimes=3)
        self.portfolio_optimizer = PortfolioOptimizerAdvanced()
        
        # Cache for performance
        self.cache = SmartCache()
        
        # Regime state
        self.current_regime = None
        self.regime_probabilities = {}
        
        logger.info("Enhanced orchestrator initialized with advanced quant features")
    
    @cached(ttl=3600)
    def analyze_market_regime(self) -> Dict[str, Any]:
        """
        Analyze current market regime.
        
        Returns:
            Regime analysis dictionary
        """
        try:
            # Get market data (use SPY as proxy)
            if "SPY" not in self.symbols:
                market_symbol = self.symbols[0] if self.symbols else "SPY"
            else:
                market_symbol = "SPY"
            
            df = self.data_fetcher.get_stock_data(market_symbol, period="1y")
            if df is None or len(df) < 50:
                return {"regime": "unknown", "confidence": 0.0}
            
            returns = df['Close'].pct_change().dropna()
            
            # Detect regime
            regimes = self.regime_detector.detect_regimes(returns, method="kmeans")
            current_regime = self.regime_detector.get_current_regime()
            regime_probs = self.regime_detector.get_regime_probabilities()
            
            self.current_regime = current_regime
            self.regime_probabilities = regime_probs
            
            # Get regime characteristics
            regime_info = self.regime_detector.regime_characteristics.get(
                current_regime, {}
            )
            
            return {
                "regime": current_regime,
                "regime_label": regime_info.get("label", "Unknown"),
                "confidence": regime_probs.get(current_regime, 0.0),
                "characteristics": regime_info,
                "probabilities": regime_probs
            }
        
        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return {"regime": "unknown", "confidence": 0.0}
    
    def generate_enhanced_signals(self, symbol: str) -> List[TradingSignal]:
        """
        Generate enhanced trading signals with factor and regime analysis.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of enhanced trading signals
        """
        # Get base signals
        base_signals = super().generate_signals(symbol)
        
        # Get market regime
        regime_analysis = self.analyze_market_regime()
        
        # Enhance signals with regime information
        enhanced_signals = []
        for signal in base_signals:
            # Adjust confidence based on regime
            regime_adjusted_confidence = signal.confidence
            
            if regime_analysis.get("regime") is not None:
                regime_label = regime_analysis.get("regime_label", "")
                
                # Adjust confidence based on regime
                if "Bull" in regime_label and signal.action == "BUY":
                    regime_adjusted_confidence = min(signal.confidence * 1.2, 1.0)
                elif "Bear" in regime_label and signal.action == "SELL":
                    regime_adjusted_confidence = min(signal.confidence * 1.2, 1.0)
                elif "Bull" in regime_label and signal.action == "SELL":
                    regime_adjusted_confidence = signal.confidence * 0.8
                elif "Bear" in regime_label and signal.action == "BUY":
                    regime_adjusted_confidence = signal.confidence * 0.8
            
            # Create enhanced signal
            enhanced_signal = TradingSignal(
                symbol=signal.symbol,
                action=signal.action,
                confidence=regime_adjusted_confidence,
                price=signal.price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                reasoning=f"{signal.reasoning} [Regime: {regime_analysis.get('regime_label', 'Unknown')}]",
                model_source=f"{signal.model_source}+regime",
                timestamp=signal.timestamp
            )
            
            enhanced_signals.append(enhanced_signal)
        
        return enhanced_signals
    
    def optimize_portfolio_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio weights using advanced methods.
        
        Args:
            returns: Asset returns DataFrame
        
        Returns:
            Optimal weights dictionary
        """
        try:
            # Use risk parity for more stable allocation
            weights = self.portfolio_optimizer.optimize_risk_parity(returns)
            return weights
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            n_assets = len(returns.columns)
            return {col: 1/n_assets for col in returns.columns}
    
    def get_factor_exposure(self, symbol: str) -> Dict[str, float]:
        """
        Get factor exposure for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Factor exposure dictionary
        """
        try:
            # Get returns for multiple symbols to build factor model
            symbols_for_factors = self.symbols[:10]  # Use top 10 symbols
            
            returns_data = {}
            for sym in symbols_for_factors:
                df = self.data_fetcher.get_stock_data(sym, period="1y")
                if df is not None and len(df) > 0:
                    returns_data[sym] = df['Close'].pct_change().dropna()
            
            if len(returns_data) < 3:
                return {}
            
            # Build factor model
            returns_df = pd.DataFrame(returns_data).dropna()
            factor_results = self.factor_model.fit(returns_df)
            
            # Get factor loadings for symbol
            if symbol in returns_df.columns:
                symbol_idx = list(returns_df.columns).index(symbol)
                loadings = self.factor_model.factor_loadings[symbol_idx]
                
                return {
                    f"Factor_{i+1}": float(loadings[i])
                    for i in range(len(loadings))
                }
            
            return {}
        
        except Exception as e:
            logger.warning(f"Factor exposure calculation failed: {e}")
            return {}
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced status with quant features."""
        base_status = super().get_status()
        
        # Add regime information
        regime_analysis = self.analyze_market_regime()
        
        enhanced_status = {
            **base_status,
            "market_regime": regime_analysis,
            "factor_model_initialized": self.factor_model.factors is not None,
            "regime_detector_initialized": self.regime_detector.regimes is not None
        }
        
        return enhanced_status
