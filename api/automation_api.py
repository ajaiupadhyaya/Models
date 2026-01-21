"""
Automated Trading Orchestration

Combines:
- Macro data fetching (FRED)
- ML predictions (ensemble + LSTM)
- AI analysis and trading recommendations
- Alpaca trading execution
- OpenAI narrative generation
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
import pandas as pd

from core.data_fetcher import DataFetcher
from core.ai_analysis import get_ai_service
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor
from core.paper_trading import AlpacaAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/automation", tags=["Automation"])

data_fetcher = DataFetcher()
ai_service = get_ai_service()

# Lazy-load Alpaca adapter
_alpaca = None

def get_alpaca_adapter():
    """Get or initialize Alpaca adapter."""
    global _alpaca
    if _alpaca is None:
        api_key = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_API_SECRET", "")
        base_url = os.getenv("ALPACA_API_BASE", "https://paper-api.alpaca.markets")
        
        if api_key and api_secret:
            try:
                _alpaca = AlpacaAdapter(api_key, api_secret, base_url)
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca adapter: {e}")
                _alpaca = False  # Mark as failed
        else:
            _alpaca = False  # Mark as not configured
    
    return _alpaca if _alpaca is not False else None


# State tracking
automated_state = {
    "last_run": None,
    "positions": {},
    "trades": [],
    "errors": [],
}


@router.get("/status")
async def automation_status():
    """Get automated trading system status."""
    return {
        "timestamp": datetime.now().isoformat(),
        "last_run": automated_state["last_run"],
        "positions_count": len(automated_state["positions"]),
        "trades_count": len(automated_state["trades"]),
        "recent_errors": automated_state["errors"][-5:] if automated_state["errors"] else []
    }


@router.post("/predict-and-trade")
async def predict_and_trade(
    symbols: str = Query("AAPL,MSFT,GOOGL", description="Comma-separated stock symbols"),
    use_lstm: bool = Query(True, description="Use LSTM model in ensemble"),
    execute_trades: bool = Query(False, description="Actually execute trades on Alpaca (paper mode)"),
    background_tasks: BackgroundTasks = None
):
    """
    Run full automated trading loop:
    1. Fetch macro data (FRED)
    2. Run ML predictions on stocks
    3. Generate AI trading recommendations
    4. Execute trades on Alpaca (if enabled)
    5. Generate narrative summary
    
    Returns:
        - predictions: Per-stock ML forecasts
        - recommendations: AI trading actions
        - trades: Executed trades (if enabled)
        - narrative: OpenAI summary of actions
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # 1. Fetch macro context
        logger.info("Fetching macro data...")
        macro_data = None
        try:
            macro_data = {
                "unemployment": data_fetcher.get_economic_indicator("UNRATE"),
                "gdp": data_fetcher.get_economic_indicator("GDP"),
                "inflation": data_fetcher.get_economic_indicator("CPIAUCSL"),
            }
            logger.info(f"Macro data: {macro_data}")
        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")
        
        # 2. Run predictions and recommendations
        predictions = {}
        recommendations = {}
        
        for symbol in symbol_list:
            try:
                logger.info(f"Processing {symbol}...")
                
                # Fetch stock data
                df = data_fetcher.get_stock_data(symbol, period="3mo")
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Ensemble prediction
                ensemble_pred = None
                try:
                    ensemble = EnsemblePredictor(lookback_window=20)
                    ensemble.train(df)
                    ensemble_pred = ensemble.predict(df)
                    if hasattr(ensemble_pred, 'iloc'):
                        ensemble_pred = ensemble_pred.iloc[-1]
                    else:
                        ensemble_pred = ensemble_pred[-1]
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed for {symbol}: {e}")
                
                # LSTM prediction (if enabled)
                lstm_pred = None
                if use_lstm and len(df) >= 20:
                    try:
                        lstm = LSTMPredictor(lookback_window=20, hidden_units=16)
                        lstm.train(df, epochs=1, verbose=0)  # 1 epoch for speed
                        lstm_pred = lstm.predict(df)
                        if hasattr(lstm_pred, 'iloc'):
                            lstm_pred = lstm_pred.iloc[-1]
                        else:
                            lstm_pred = lstm_pred[-1]
                    except Exception as e:
                        logger.warning(f"LSTM prediction failed for {symbol}: {e}")
                
                # Average predictions
                predictions_list = [p for p in [ensemble_pred, lstm_pred] if p is not None]
                avg_prediction = sum(predictions_list) / len(predictions_list) if predictions_list else current_price
                confidence = 0.65 if len(predictions_list) > 1 else 0.5
                
                predictions[symbol] = {
                    "current_price": float(current_price),
                    "ensemble_prediction": float(ensemble_pred) if ensemble_pred else None,
                    "lstm_prediction": float(lstm_pred) if lstm_pred else None,
                    "avg_prediction": float(avg_prediction),
                    "implied_move_pct": float(((avg_prediction - current_price) / current_price * 100))
                }
                
                # 3. AI trading recommendation
                market_context = f"Macro: Unemployment={macro_data.get('unemployment', 'N/A')}" if macro_data else "Normal conditions"
                
                insight = ai_service.generate_trading_insight(
                    symbol=symbol,
                    current_price=current_price,
                    prediction=avg_prediction,
                    confidence=confidence,
                    market_context=market_context
                )
                
                recommendations[symbol] = insight
                
                logger.info(f"{symbol}: {insight.get('action')} ({insight.get('reasoning', '')[:50]}...)")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                automated_state["errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "error": str(e)
                })
        
        # 4. Execute trades (if enabled)
        executed_trades = []
        alpaca = get_alpaca_adapter()
        if execute_trades and alpaca and alpaca.is_authenticated():
            logger.info("Executing trades on Alpaca...")
            for symbol, rec in recommendations.items():
                try:
                    action = rec.get("action", "HOLD")
                    if action == "BUY":
                        # Example: buy 10 shares
                        order = alpaca.submit_order(
                            symbol=symbol,
                            qty=10,
                            side="buy",
                            type="market"
                        )
                        if order:
                            executed_trades.append({
                                "symbol": symbol,
                                "action": "BUY",
                                "qty": 10,
                                "order_id": order.get("id", "unknown"),
                                "status": "submitted"
                            })
                            logger.info(f"BUY order submitted for {symbol}")
                    
                    elif action == "SELL":
                        # Example: sell 10 shares (if we have them)
                        positions = alpaca.get_positions()
                        if positions and symbol in positions:
                            qty = positions[symbol]["qty"]
                            order = alpaca.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side="sell",
                                type="market"
                            )
                            if order:
                                executed_trades.append({
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "qty": qty,
                                    "order_id": order.get("id", "unknown"),
                                    "status": "submitted"
                                })
                                logger.info(f"SELL order submitted for {symbol}")
                
                except Exception as e:
                    logger.error(f"Trade execution failed for {symbol}: {e}")
                    automated_state["errors"].append({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "error": f"Trade execution: {str(e)}"
                    })
        
        # 5. Generate narrative
        narrative_prompt = f"""
        Automated trading run completed at {datetime.now().isoformat()}
        
        Symbols analyzed: {', '.join(symbol_list)}
        
        Predictions summary:
        {chr(10).join([f"  - {sym}: {data.get('avg_prediction', 'N/A')} (move: {data.get('implied_move_pct', 0):+.2f}%)" for sym, data in predictions.items()])}
        
        Recommendations:
        {chr(10).join([f"  - {sym}: {rec.get('action', 'HOLD')} ({rec.get('reasoning', '')[:80]})" for sym, rec in recommendations.items()])}
        
        Trades executed: {len(executed_trades)}
        """
        
        narrative = ai_service.explain_metrics({
            "predictions_generated": len(predictions),
            "recommendations_made": len(recommendations),
            "trades_executed": len(executed_trades),
            "confidence": "medium"
        })
        
        # Update state
        automated_state["last_run"] = datetime.now().isoformat()
        automated_state["positions"] = alpaca.get_positions() if execute_trades else {}
        automated_state["trades"].extend(executed_trades)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "predictions": predictions,
            "recommendations": recommendations,
            "executed_trades": executed_trades,
            "narrative": narrative,
            "macro_context": macro_data
        }
    
    except Exception as e:
        logger.error(f"Automation error: {e}")
        automated_state["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_current_positions():
    """Get current Alpaca positions."""
    try:
        alpaca = get_alpaca_adapter()
        if not alpaca:
            return {"error": "Alpaca not configured", "positions": {}, "count": 0}
        
        positions = alpaca.get_positions()
        return {
            "timestamp": datetime.now().isoformat(),
            "positions": positions or {},
            "count": len(positions) if positions else 0
        }
    except Exception as e:
        logger.error(f"Position fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account")
async def get_account_info():
    """Get Alpaca account status."""
    try:
        alpaca = get_alpaca_adapter()
        if not alpaca:
            return {"error": "Alpaca not configured", "account": {}}
        
        account = alpaca.get_account_status()
        return {
            "timestamp": datetime.now().isoformat(),
            "account": account or {"error": "Account unavailable"}
        }
    except Exception as e:
        logger.error(f"Account fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
