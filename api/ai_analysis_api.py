"""
AI Analysis API Endpoints

- /api/v1/ai/market-summary — Analyze current market trends
- /api/v1/ai/stock-analysis — Detailed stock analysis and recommendation
- /api/v1/ai/trading-insight — AI-powered trading recommendation
- /api/v1/ai/sentiment — Sentiment analysis of market text
- /api/v1/ai/metrics-explained — Explain financial metrics
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from core.ai_analysis import get_ai_service
from core.data_fetcher import DataFetcher
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["AI Analysis"])

data_fetcher = DataFetcher()
ai_service = get_ai_service()


@router.get("/market-summary")
async def market_summary(symbols: str = Query("AAPL,MSFT,GOOGL", description="Comma-separated tickers")):
    """
    Analyze multiple stocks and provide AI-powered market summary.
    Cached 5 minutes per symbol set (see api/cache.py).
    """
    from api.cache import get_cached, set_cached, cache_key, CACHE_TTL_MARKET_SUMMARY
    key = cache_key("ai", "market-summary", symbols)
    cached = get_cached(key)
    if cached is not None:
        return cached
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        analyses = {}
        
        for symbol in symbol_list:
            try:
                # Fetch recent data
                df = data_fetcher.get_stock_data(symbol, period="1mo")
                if df is None or df.empty:
                    analyses[symbol] = {"error": "Data unavailable"}
                    continue
                
                # Prepare metrics
                metrics = {
                    "RSI": 50.0,  # Placeholder; would compute real RSI
                    "MACD": 0.0,
                    "SMA_50": df['Close'].tail(50).mean() if len(df) >= 50 else df['Close'].mean(),
                    "Volume_Trend": "up" if df['Volume'].iloc[-1] > df['Volume'].mean() else "down"
                }
                
                # AI analysis
                analysis = ai_service.analyze_price_chart(symbol, df, metrics)
                analyses[symbol] = {
                    "price": df['Close'].iloc[-1],
                    "analysis": analysis
                }
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                analyses[symbol] = {"error": str(e)}
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "analyses": analyses,
            "market_tone": "Neutral - Run sentiment analysis for more"
        }
        set_cached(key, result, CACHE_TTL_MARKET_SUMMARY)
        return result
    except Exception as e:
        logger.error(f"Market summary error: {e}")
        # Return 200 with placeholder so frontend stays operational
        symbol_list = [s.strip().upper() for s in symbols.split(",")] if symbols else ["AAPL"]
        analyses = {sym: {"price": 0.0, "analysis": "Data temporarily unavailable. Check API and network."} for sym in symbol_list}
        return {
            "timestamp": datetime.now().isoformat(),
            "analyses": analyses,
            "market_tone": "Unavailable - check API and data sources",
        }


@router.get("/stock-analysis/{symbol}")
async def stock_analysis(
    symbol: str,
    include_prediction: bool = Query(True, description="Include ML prediction")
):
    """
    Deep-dive AI analysis of a single stock.
    
    Returns:
        - current_price: Latest price
        - technical_analysis: AI chart analysis
        - prediction: ML model forecast (optional)
        - trading_insight: AI recommendation
        - sentiment: Market sentiment for this stock
    """
    try:
        symbol = symbol.upper()
        
        # Fetch data
        df = data_fetcher.get_stock_data(symbol, period="3mo")
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        current_price = df['Close'].iloc[-1]
        
        # Technical analysis via AI
        technical_analysis = ai_service.analyze_price_chart(symbol, df)
        
        # ML prediction (optional)
        prediction_data = None
        if include_prediction and len(df) >= 20:
            try:
                # Train ensemble model on recent data
                model = EnsemblePredictor(lookback_window=20)
                model.train(df)
                next_price = model.predict(df).iloc[-1] if hasattr(model.predict(df), 'iloc') else model.predict(df)[-1]
                confidence = 0.65  # Reasonable default confidence
                
                prediction_data = {
                    "next_price": float(next_price),
                    "implied_change_pct": float(((next_price - current_price) / current_price * 100))
                }
            except Exception as e:
                logger.warning(f"Prediction error for {symbol}: {e}")
        
        # Trading insight
        insight = ai_service.generate_trading_insight(
            symbol=symbol,
            current_price=current_price,
            prediction=prediction_data["next_price"] if prediction_data else current_price,
            confidence=0.65 if prediction_data else 0.0,
            market_context="Normal market conditions"
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "current_price": current_price,
            "technical_analysis": technical_analysis,
            "prediction": prediction_data,
            "trading_insight": insight
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock analysis error: {e}")
        # Return 200 with minimal structure so frontend stays operational
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol.upper(),
            "current_price": 0.0,
            "technical_analysis": "Analysis temporarily unavailable. Ensure API is running and OPENAI_API_KEY is set for full analysis.",
            "prediction": None,
            "trading_insight": {"reasoning": "Unable to generate insight. Check API and configuration.", "recommendation": "—"},
        }


@router.post("/trading-insight")
async def trading_insight(
    symbol: str = Query(...),
    current_price: float = Query(...),
    predicted_price: float = Query(...),
    confidence: float = Query(0.65, ge=0, le=1)
):
    """
    Get AI trading recommendation based on ML prediction.
    
    Returns:
        - action: BUY, SELL, or HOLD
        - reasoning: AI explanation
        - risk_level: low, medium, high
        - stop_loss_pct: Recommended stop loss
        - take_profit_pct: Recommended take profit
    """
    try:
        insight = ai_service.generate_trading_insight(
            symbol=symbol.upper(),
            current_price=current_price,
            prediction=predicted_price,
            confidence=confidence
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol.upper(),
            **insight
        }
    
    except Exception as e:
        logger.error(f"Trading insight error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment")
async def analyze_sentiment(text: str = Query(...)):
    """
    Analyze sentiment of market news or analysis text.
    
    Returns:
        - sentiment: bullish, bearish, or neutral
        - score: -1.0 to 1.0
        - reasoning: Explanation
    """
    try:
        result = ai_service.sentiment_analysis(text)
        return {
            "timestamp": datetime.now().isoformat(),
            "input": text[:100] + "..." if len(text) > 100 else text,
            **result
        }
    
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-metrics")
async def explain_metrics(metrics: dict):
    """
    Explain financial metrics in plain English.
    
    Request:
        {"Sharpe Ratio": 1.5, "Sortino Ratio": 2.1, "Max Drawdown": -0.15}
    
    Returns:
        - explanation: Plain-English description
    """
    try:
        explanation = ai_service.explain_metrics(metrics)
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Metric explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nl-query")
async def nl_query(q: str = Query(..., description="Natural language question (e.g. What's driving tech stocks today?)")):
    """
    Answer a natural-language question using macro data, market summary, and LLM.
    Uses cached macro and market-summary when available.
    """
    from api.cache import get_cached, cache_key
    context_parts = []
    try:
        macro = get_cached(cache_key("data", "macro"))
        if macro and isinstance(macro, dict) and "series" in macro:
            for s in macro.get("series", [])[:6]:
                desc = s.get("description", "")
                data = s.get("data", [])
                if data:
                    latest = data[-1]
                    context_parts.append(f"- {desc}: {latest.get('value', 'N/A')} (as of {latest.get('date', '')})")
        if not context_parts:
            context_parts.append("- Macro: (no cached data; run Economic panel to populate)")
    except Exception as e:
        context_parts.append(f"- Macro: (unavailable: {e})")
    try:
        market = get_cached(cache_key("ai", "market-summary", "AAPL,MSFT,GOOGL,SPY"))
        if market and isinstance(market, dict) and "analyses" in market:
            context_parts.append("- Market summary (AAPL, MSFT, GOOGL, SPY):")
            for sym, v in list((market.get("analyses") or {}).items())[:4]:
                if isinstance(v, dict) and v.get("analysis"):
                    context_parts.append(f"  {sym}: {str(v.get('analysis', ''))[:120]}...")
    except Exception:
        pass
    context = "\n".join(context_parts) if context_parts else "No macro or market data cached yet."
    answer = ai_service.answer_nl_query(context, q)
    return {"answer": answer, "question": q}
