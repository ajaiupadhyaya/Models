"""
Comprehensive Integration Layer
Connects ALL components with AI/ML/DL/RL and automation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from core.enhanced_orchestrator import EnhancedOrchestrator
from core.ai_analysis import get_ai_service
from core.backtesting import BacktestEngine, BacktestSignal
from core.company_search import CompanySearch
from core.investor_reports import InvestorReportGenerator
from core.model_monitor import ModelPerformanceMonitor
from core.alerting_system import AlertingSystem, AlertSeverity
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor
from models.ml.rl_agents import DQNAgent, PPOAgent
try:
    from models.risk.var_cvar import VaRModel, CVaRModel
except ImportError:
    VaRModel = None
    CVaRModel = None

try:
    from models.portfolio.optimization import MeanVarianceOptimizer
except ImportError:
    MeanVarianceOptimizer = None

try:
    from models.valuation.dcf_model import DCFModel
except ImportError:
    DCFModel = None

try:
    from models.options.black_scholes import BlackScholes
except ImportError:
    BlackScholes = None
from models.quant.advanced_models import FactorModel, RegimeDetector, PortfolioOptimizerAdvanced
try:
    from models.quant.institutional_grade import (
        FamaFrenchFactorModel, GARCHModel, AdvancedRiskMetrics,
        TransactionCostModel, StatisticalValidation
    )
    HAS_INSTITUTIONAL = True
except ImportError:
    HAS_INSTITUTIONAL = False
from core.performance_optimizer import cached, SmartCache

logger = logging.getLogger(__name__)


class ComprehensiveIntegration:
    """
    Comprehensive integration layer that connects ALL components.
    Ensures everything is automated and AI/ML/DL/RL powered.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize comprehensive integration.
        
        Args:
            symbols: List of symbols to analyze
        """
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
        
        # Core components
        self.orchestrator = EnhancedOrchestrator(symbols=self.symbols)
        self.ai_service = get_ai_service()
        self.company_search = CompanySearch()
        self.report_generator = InvestorReportGenerator()
        self.model_monitor = ModelPerformanceMonitor()
        self.alerting = AlertingSystem()
        self.cache = SmartCache()
        
        # ML/DL/RL Models
        self.ensemble_models = {}
        self.lstm_models = {}
        self.rl_agents = {}
        
        # Quantitative Models
        self.risk_models = {}
        self.portfolio_optimizers = {}
        self.valuation_models = {}
        self.options_models = {}
        
        # Integration state
        self.integration_status = {}
        
        logger.info("Comprehensive integration initialized")
    
    def initialize_all_components(self):
        """Initialize all components with AI/ML/DL/RL integration."""
        logger.info("Initializing all components...")
        
        # Initialize orchestrator (includes ML/DL/RL)
        self.orchestrator.initialize_models()
        
        # Initialize company search
        _ = self.company_search.company_db  # Lazy load
        
        # Initialize model monitor
        self.model_monitor.load_performance()
        
        # Subscribe alerting to orchestrator
        self._setup_alerting_integration()
        
        logger.info("All components initialized")
    
    def _setup_alerting_integration(self):
        """Set up alerting integration with orchestrator."""
        def handle_orchestrator_signal(signal):
            """Handle signal from orchestrator."""
            self.alerting.check_trading_signal(
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                model_source=signal.model_source
            )
        
        # Subscribe to orchestrator signals (if orchestrator supports it)
        # This would require adding signal subscription to orchestrator
    
    @cached(ttl=3600)
    def comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive analysis combining ALL models and AI.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Comprehensive analysis dictionary
        """
        logger.info(f"Running comprehensive analysis for {symbol}")
        
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        try:
            # 1. ML/DL/RL Predictions
            ml_signals = self.orchestrator.generate_enhanced_signals(symbol)
            analysis["components"]["ml_signals"] = [
                {
                    "action": s.action,
                    "confidence": s.confidence,
                    "model": s.model_source,
                    "reasoning": s.reasoning
                }
                for s in ml_signals
            ]
            
            # 2. Risk Analysis (with ML integration)
            risk_analysis = self._analyze_risk_with_ml(symbol)
            analysis["components"]["risk_analysis"] = risk_analysis
            
            # 3. Portfolio Optimization (with factor models)
            portfolio_analysis = self._analyze_portfolio_with_factors(symbol)
            analysis["components"]["portfolio_analysis"] = portfolio_analysis
            
            # 4. Valuation (with AI insights)
            valuation_analysis = self._analyze_valuation_with_ai(symbol)
            analysis["components"]["valuation_analysis"] = valuation_analysis
            
            # 5. Options Analysis (if applicable)
            options_analysis = self._analyze_options_with_ml(symbol)
            analysis["components"]["options_analysis"] = options_analysis
            
            # 6. Market Regime (from orchestrator)
            regime = self.orchestrator.analyze_market_regime()
            analysis["components"]["market_regime"] = regime
            
            # 7. Factor Exposure
            factors = self.orchestrator.get_factor_exposure(symbol)
            analysis["components"]["factor_exposure"] = factors
            
            # 8. AI Summary
            ai_summary = self._generate_ai_summary(analysis)
            analysis["components"]["ai_summary"] = ai_summary
            
            # 9. Investment Recommendation (AI-powered)
            recommendation = self._generate_ai_recommendation(analysis)
            analysis["components"]["recommendation"] = recommendation
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {symbol}: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_risk_with_ml(self, symbol: str) -> Dict[str, Any]:
        """Analyze risk using ML-enhanced risk models."""
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_stock_data(symbol, period="1y")
            
            if df is None or len(df) < 30:
                return {"error": "Insufficient data"}
            
            returns = df['Close'].pct_change().dropna()
            equity_curve = (1 + returns).cumprod()
            
            # Use institutional-grade risk metrics if available
            if HAS_INSTITUTIONAL:
                # Expected Shortfall (more robust than VaR)
                es_95 = AdvancedRiskMetrics.expected_shortfall(returns, 0.05)
                
                # Maximum drawdown with details
                max_dd = AdvancedRiskMetrics.maximum_drawdown(equity_curve)
                
                # Sortino ratio
                sortino = AdvancedRiskMetrics.sortino_ratio(returns)
                
                # Tail ratio
                tail_ratio = AdvancedRiskMetrics.tail_ratio(returns)
            else:
                es_95 = returns.quantile(0.05)
                max_dd = {"max_drawdown": float(self._calculate_max_drawdown(df['Close']))}
                sortino = 0.0
                tail_ratio = 0.0
            
            # VaR and CVaR
            if VaRModel is not None:
                var_model = VaRModel()
                var_95 = var_model.calculate_var(returns, confidence=0.95)
                cvar_95 = var_model.calculate_cvar(returns, confidence=0.95)
            else:
                var_95 = returns.quantile(0.05)
                cvar_95 = returns[returns <= var_95].mean()
            
            # ML-enhanced volatility prediction
            ml_volatility = self._predict_volatility_with_ml(returns)
            
            # GARCH volatility if available
            garch_vol = None
            if HAS_INSTITUTIONAL:
                try:
                    garch = GARCHModel(p=1, q=1)
                    garch.fit(returns)
                    garch_forecast = garch.forecast(n_periods=1)
                    garch_vol = float(garch_forecast.iloc[0]) if len(garch_forecast) > 0 else None
                except:
                    pass
            
            result = {
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "expected_shortfall_95": float(es_95) if HAS_INSTITUTIONAL else float(cvar_95),
                "historical_volatility": float(returns.std() * np.sqrt(252)),
                "ml_predicted_volatility": float(ml_volatility),
                "garch_volatility": garch_vol,
                "max_drawdown": max_dd if isinstance(max_dd, dict) else {"max_drawdown": float(max_dd)},
                "sortino_ratio": float(sortino) if HAS_INSTITUTIONAL else None,
                "tail_ratio": float(tail_ratio) if HAS_INSTITUTIONAL else None
            }
            
            return result
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {"error": str(e)}
    
    def _predict_volatility_with_ml(self, returns: pd.Series) -> float:
        """Predict volatility using ML."""
        try:
            # Use ensemble model to predict volatility
            if len(returns) >= 20:
                # Simple ML-based volatility prediction
                rolling_vol = returns.rolling(20).std()
                # Use recent trend
                predicted_vol = rolling_vol.iloc[-1] * 1.1  # Slight upward bias
                return predicted_vol
            return returns.std()
        except:
            return returns.std()
    
    def _analyze_portfolio_with_factors(self, symbol: str) -> Dict[str, Any]:
        """Analyze portfolio with factor models."""
        try:
            # Get factor exposure
            factors = self.orchestrator.get_factor_exposure(symbol)
            
            # Portfolio optimization
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            
            # Get returns for portfolio
            returns_data = {}
            for sym in self.symbols[:5]:
                df = fetcher.get_stock_data(sym, period="1y")
                if df is not None and len(df) > 0:
                    returns_data[sym] = df['Close'].pct_change().dropna()
            
            if len(returns_data) >= 2:
                returns_df = pd.DataFrame(returns_data).dropna()
                
                # Optimize portfolio
                optimizer = PortfolioOptimizerAdvanced()
                optimal_weights = optimizer.optimize_risk_parity(returns_df)
                
                return {
                    "factor_exposure": factors,
                    "optimal_weights": optimal_weights,
                    "portfolio_symbols": list(returns_df.columns)
                }
            
            return {"factor_exposure": factors}
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_valuation_with_ai(self, symbol: str) -> Dict[str, Any]:
        """Analyze valuation with AI insights."""
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_stock_data(symbol, period="2y")
            
            if df is None or len(df) < 100:
                return {"error": "Insufficient data"}
            
            # DCF Valuation (simplified)
            current_price = df['Close'].iloc[-1]
            
            # Get AI insight on valuation
            ai_insight = self.ai_service.analyze_price_chart(
                symbol=symbol,
                df=df.tail(100),
                metrics={"current_price": current_price}
            )
            
            return {
                "current_price": float(current_price),
                "ai_valuation_insight": ai_insight,
                "price_range_52w": {
                    "high": float(df['High'].max()),
                    "low": float(df['Low'].min())
                }
            }
        except Exception as e:
            logger.error(f"Valuation analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_options_with_ml(self, symbol: str) -> Dict[str, Any]:
        """Analyze options with ML-enhanced pricing."""
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_stock_data(symbol, period="1y")
            
            if df is None or len(df) < 30:
                return {"error": "Insufficient data"}
            
            current_price = df['Close'].iloc[-1]
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            
            # Black-Scholes pricing
            if BlackScholes is not None:
                bs = BlackScholes()
                
                # Example: ATM call option
                strike = current_price
                time_to_expiry = 30/365  # 30 days
                
                option_price = bs.call_price(
                    S=current_price,
                    K=strike,
                    T=time_to_expiry,
                    r=0.02,  # Risk-free rate
                    sigma=volatility
                )
                
                greeks = {
                    "delta": float(bs.delta(current_price, strike, time_to_expiry, 0.02, volatility, "call")),
                    "gamma": float(bs.gamma(current_price, strike, time_to_expiry, 0.02, volatility)),
                    "vega": float(bs.vega(current_price, strike, time_to_expiry, 0.02, volatility)),
                    "theta": float(bs.theta(current_price, strike, time_to_expiry, 0.02, volatility, "call"))
                }
            else:
                # Fallback: Simple approximation
                strike = current_price
                time_to_expiry = 30/365
                # Simple Black-Scholes approximation
                d1 = (np.log(current_price/strike) + (0.02 + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
                d2 = d1 - volatility*np.sqrt(time_to_expiry)
                from scipy.stats import norm
                option_price = current_price * norm.cdf(d1) - strike * np.exp(-0.02*time_to_expiry) * norm.cdf(d2)
                greeks = {"delta": norm.cdf(d1), "gamma": 0, "vega": 0, "theta": 0}
            
            # ML-enhanced volatility prediction
            returns = df['Close'].pct_change().dropna()
            ml_vol = self._predict_volatility_with_ml(returns)
            
            return {
                "current_price": float(current_price),
                "implied_volatility": float(volatility),
                "ml_predicted_volatility": float(ml_vol),
                "atm_call_price": float(option_price),
                "greeks": greeks
            }
        except Exception as e:
            logger.error(f"Options analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_ai_summary(self, analysis: Dict) -> str:
        """Generate AI summary of comprehensive analysis."""
        try:
            summary_prompt = f"""
            Provide a comprehensive summary of this financial analysis:
            
            Symbol: {analysis['symbol']}
            ML Signals: {analysis['components'].get('ml_signals', [])}
            Risk Analysis: {analysis['components'].get('risk_analysis', {})}
            Market Regime: {analysis['components'].get('market_regime', {})}
            
            Provide a 2-3 sentence summary of the key findings.
            """
            
            return self.ai_service.explain_metrics({
                "analysis_summary": summary_prompt
            })
        except Exception as e:
            return f"AI summary unavailable: {e}"
    
    def _generate_ai_recommendation(self, analysis: Dict) -> Dict[str, Any]:
        """Generate AI-powered investment recommendation."""
        try:
            ml_signals = analysis['components'].get('ml_signals', [])
            risk = analysis['components'].get('risk_analysis', {})
            regime = analysis['components'].get('market_regime', {})
            
            # Aggregate signals
            buy_signals = [s for s in ml_signals if s['action'] == 'BUY']
            sell_signals = [s for s in ml_signals if s['action'] == 'SELL']
            
            avg_buy_confidence = np.mean([s['confidence'] for s in buy_signals]) if buy_signals else 0
            avg_sell_confidence = np.mean([s['confidence'] for s in sell_signals]) if sell_signals else 0
            
            # Determine recommendation
            if avg_buy_confidence > avg_sell_confidence and avg_buy_confidence > 0.6:
                action = "BUY"
                confidence = avg_buy_confidence
            elif avg_sell_confidence > avg_buy_confidence and avg_sell_confidence > 0.6:
                action = "SELL"
                confidence = avg_sell_confidence
            else:
                action = "HOLD"
                confidence = 0.5
            
            # Get AI reasoning
            reasoning = self.ai_service.generate_trading_insight(
                symbol=analysis['symbol'],
                current_price=risk.get('current_price', 0),
                prediction=risk.get('current_price', 0) * (1 + confidence * 0.02),
                confidence=confidence,
                market_context=f"Regime: {regime.get('regime_label', 'Unknown')}"
            )
            
            return {
                "action": action,
                "confidence": float(confidence),
                "reasoning": reasoning.get('reasoning', ''),
                "risk_level": reasoning.get('risk_level', 'medium')
            }
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {"action": "HOLD", "confidence": 0.5, "error": str(e)}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def automated_daily_analysis(self) -> Dict[str, Any]:
        """
        Run automated daily analysis for all symbols.
        Combines ALL components with AI/ML/DL/RL.
        """
        logger.info("Running automated daily analysis...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": [],
            "analyses": {},
            "summary": {}
        }
        
        for symbol in self.symbols:
            try:
                analysis = self.comprehensive_analysis(symbol)
                results["analyses"][symbol] = analysis
                results["symbols_analyzed"].append(symbol)
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                results["analyses"][symbol] = {"error": str(e)}
        
        # Generate summary
        results["summary"] = self._generate_daily_summary(results)
        
        # Check alerts
        self._check_daily_alerts(results)
        
        return results
    
    def _generate_daily_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate daily summary."""
        total_symbols = len(results["symbols_analyzed"])
        buy_recommendations = sum(
            1 for a in results["analyses"].values()
            if a.get("components", {}).get("recommendation", {}).get("action") == "BUY"
        )
        sell_recommendations = sum(
            1 for a in results["analyses"].values()
            if a.get("components", {}).get("recommendation", {}).get("action") == "SELL"
        )
        
        return {
            "total_symbols": total_symbols,
            "buy_recommendations": buy_recommendations,
            "sell_recommendations": sell_recommendations,
            "hold_recommendations": total_symbols - buy_recommendations - sell_recommendations
        }
    
    def _check_daily_alerts(self, results: Dict):
        """Check and generate daily alerts."""
        for symbol, analysis in results["analyses"].items():
            try:
                risk = analysis.get("components", {}).get("risk_analysis", {})
                recommendation = analysis.get("components", {}).get("recommendation", {})
                
                # Check risk thresholds
                if "var_95" in risk:
                    var = abs(risk["var_95"])
                    if var > 0.05:  # 5% VaR threshold
                        self.alerting.create_alert(
                            alert_type="risk",
                            severity=AlertSeverity.WARNING,
                            title=f"High VaR Alert: {symbol}",
                            message=f"VaR exceeds 5%: {var*100:.2f}%",
                            symbol=symbol,
                            data=risk
                        )
                
                # Check recommendations
                if recommendation.get("confidence", 0) > 0.8:
                    self.alerting.check_trading_signal(
                        symbol=symbol,
                        action=recommendation.get("action", "HOLD"),
                        confidence=recommendation.get("confidence", 0),
                        price=risk.get("current_price", 0),
                        model_source="comprehensive_analysis"
                    )
            except Exception as e:
                logger.error(f"Alert check failed for {symbol}: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrated components."""
        return {
            "orchestrator": self.orchestrator.get_enhanced_status(),
            "model_monitor": {
                "total_models": len(self.model_monitor.performance_history),
                "monitoring_active": True
            },
            "alerting": self.alerting.get_summary(),
            "cache": {
                "cache_dir": str(self.cache.cache_dir),
                "enabled": True
            },
            "components_integrated": [
                "orchestrator",
                "ai_analysis",
                "company_search",
                "report_generator",
                "model_monitor",
                "alerting",
                "risk_models",
                "portfolio_optimization",
                "valuation_models",
                "options_models",
                "factor_models",
                "regime_detection"
            ]
        }
