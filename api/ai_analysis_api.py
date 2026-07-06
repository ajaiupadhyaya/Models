"""
AI Analysis API Endpoints

- /api/v1/ai/market-summary — Analyze current market trends
- /api/v1/ai/stock-analysis — Detailed stock analysis and recommendation
- /api/v1/ai/trading-insight — AI-powered trading recommendation
- /api/v1/ai/sentiment — Sentiment analysis of market text
- /api/v1/ai/metrics-explained — Explain financial metrics
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from core.ai_analysis import get_ai_service
from core.data_fetcher import DataFetcher
from models.ml.advanced_trading import EnsemblePredictor

logger = logging.getLogger(__name__)

try:
    from api.auth_api import get_current_user_if_configured
    _auth_deps = [Depends(get_current_user_if_configured)]
except Exception:
    _auth_deps = []

router = APIRouter(tags=["AI Analysis"], dependencies=_auth_deps)

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
        
        # ML prediction (optional); chart_overlay for D3 prediction band
        prediction_data = None
        chart_overlay = None
        if include_prediction and len(df) >= 20:
            try:
                # Train ensemble model on recent data
                model = EnsemblePredictor(lookback_window=20)
                model.train(df)
                next_price = model.predict(df).iloc[-1] if hasattr(model.predict(df), 'iloc') else model.predict(df)[-1]
                
                low = float(next_price * 0.97)
                high = float(next_price * 1.03)
                prediction_data = {
                    "next_price": float(next_price),
                    "implied_change_pct": float(((next_price - current_price) / current_price * 100)),
                    "confidence": 0.65,
                    "confidence_interval_low": low,
                    "confidence_interval_high": high,
                }
                chart_overlay = {
                    "next_price": float(next_price),
                    "low": low,
                    "high": high,
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
        
        # Sentiment from technical analysis snippet (optional)
        sentiment_data = None
        if technical_analysis and len(technical_analysis) > 20:
            try:
                sent = ai_service.sentiment_analysis(technical_analysis[:500])
                sentiment_data = {
                    "score": sent.get("score", 0.0),
                    "label": sent.get("sentiment", "neutral"),
                    "reasoning": sent.get("reasoning", "")[:200],
                }
            except Exception:
                pass
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "current_price": current_price,
            "technical_analysis": technical_analysis,
            "prediction": prediction_data,
            "trading_insight": insight,
            "sentiment": sentiment_data,
        }
        if chart_overlay is not None:
            result["chart_overlay"] = chart_overlay
        return result
    
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
            "sentiment": None,
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


from pydantic import BaseModel as PydanticBaseModel


class SummarizeRequest(PydanticBaseModel):
    """Request body for text summarization."""
    text: str


class ChatRequest(PydanticBaseModel):
    """Request body for AI Research Assistant chat (Claude)."""
    message: str
    history: Optional[List[Dict[str, str]]] = None  # [{role, content}]


CLAUDE_TOOLS = [
    {
        "name": "run_dcf",
        "description": "Run DCF valuation for a stock. Returns intrinsic value per share, current price, upside/downside %, sensitivity table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker (e.g. AAPL)"},
                "wacc": {"type": "number", "description": "WACC as decimal (e.g. 0.10 for 10%)"},
                "terminal_growth_rate": {"type": "number", "description": "Terminal growth rate (e.g. 0.03 for 3%)"},
            },
            "required": ["symbol", "wacc", "terminal_growth_rate"],
        },
    },
    {
        "name": "screen_stocks",
        "description": "Run stock screener with filters. Returns top 10 matching tickers with symbol, name, sector, market_cap, P/E, P/B.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {"type": "string", "description": "Filter by sector (e.g. Technology, Healthcare)"},
                "min_market_cap": {"type": "number", "description": "Minimum market cap in USD"},
                "max_pe": {"type": "number", "description": "Maximum P/E ratio"},
                "min_revenue_growth": {"type": "number", "description": "Minimum revenue growth (optional)"},
                "max_debt_equity": {"type": "number", "description": "Maximum debt/equity ratio"},
            },
            "required": [],
        },
    },
    {
        "name": "get_company_overview",
        "description": "Get company overview: price, market cap, P/E, EV/EBITDA, revenue, EBITDA, sector, description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "run_backtest",
        "description": "Run backtest for a strategy. Returns Sharpe, CAGR, max drawdown, win rate, alpha/beta vs SPY.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "description": "sma_cross, rsi_mean_reversion, or factor_momentum"},
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of tickers (first used)"},
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
            },
            "required": ["strategy", "tickers", "start_date", "end_date"],
        },
    },
    {
        "name": "get_macro_snapshot",
        "description": "Get latest macro indicators: CPI, unemployment, Fed funds rate, 10Y yield, GDP from DB/FRED.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _run_data_tools(message: str) -> str:
    """Run DCF, screener, or profile based on message keywords; return context string."""
    import re
    context_parts = []
    msg_lower = message.lower()
    # DCF: e.g. "DCF value of AAPL at 9% WACC"
    dcf_match = re.search(r"dcf\s+(?:value\s+)?(?:of\s+)?([a-z]{1,5})\s*(?:at\s+)?(\d+(?:\.\d+)?)\s*%\s*wacc", msg_lower, re.I)
    if dcf_match or ("dcf" in msg_lower and "wacc" in msg_lower):
        ticker = "AAPL"
        wacc = 0.10
        if dcf_match:
            ticker = dcf_match.group(1).upper()
            wacc = float(dcf_match.group(2)) / 100.0
        try:
            from api.equity_api import compute_dcf
            result = compute_dcf(ticker, wacc, 0.03)
            if isinstance(result, dict) and "error" not in result:
                context_parts.append(f"DCF result for {ticker}: intrinsic value ${result.get('intrinsic_value_per_share')} per share; current price {result.get('current_price')}; upside {result.get('upside_downside_pct')}%.")
        except Exception as e:
            context_parts.append(f"DCF lookup failed: {e}")
    # Screener: "profitable small-cap" or "low debt industrials"
    if "screener" in msg_lower or "small-cap" in msg_lower or "low debt" in msg_lower or "find me" in msg_lower:
        try:
            from core.company_search import CompanySearch
            searcher = CompanySearch()
            companies = searcher.get_top_companies(30) if "small" not in msg_lower else [c for c in searcher.get_top_companies(50) if (c.get("market_cap") or 0) < 2e9]
            if companies:
                context_parts.append("Screener results: " + "; ".join(
                    f"{c.get('ticker') or c.get('symbol')} ({c.get('name', '')[:20]})" for c in companies[:10]
                ))
        except Exception as e:
            context_parts.append(f"Screener failed: {e}")
    # Company profile
    ticker_match = re.search(r"\b([A-Z]{1,5})\b", message)
    if ticker_match and ("profile" in msg_lower or "overview" in msg_lower or "what is" in msg_lower):
        sym = ticker_match.group(1).upper()
        try:
            from core.db import get_company_profile
            prof = get_company_profile(sym)
            if prof:
                context_parts.append(f"Company {sym}: {prof.get('name')}; sector {prof.get('sector')}; industry {prof.get('industry')}; market_cap {prof.get('market_cap')}.")
        except Exception as e:
            context_parts.append(f"Profile lookup failed: {e}")
    return "\n".join(context_parts) if context_parts else ""


@router.post("/chat")
async def ai_research_chat(body: ChatRequest):
    """
    AI Research Assistant: chat with Claude using tool-use (DCF, screener, company overview, backtest, macro).
    Claude invokes tools; we execute them and return results. Supports multi-turn with history.
    """
    import json
    import os
    from api.ai_tools import execute_tool

    if not body.message or not body.message.strip():
        return {"reply": "", "error": "Empty message", "tool_calls": [], "structured_data": {}}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not api_key.strip():
        return {
            "reply": "ANTHROPIC_API_KEY not configured. Set it in .env for the AI Research Assistant.",
            "tool_calls": [],
            "structured_data": {},
        }

    system = (
        "You are a financial research assistant with access to tools. "
        "When the user asks for a DCF, stock screen, company overview, backtest, or macro snapshot, use the appropriate tool. "
        "Present the tool results clearly in your response. Do not invent data."
    )

    messages: List[Dict[str, Any]] = []
    if body.history:
        for h in body.history[-10:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": body.message})

    tool_calls_log: List[Dict[str, Any]] = []
    structured_data: Dict[str, Any] = {}
    max_rounds = 5
    round_num = 0

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        while round_num < max_rounds:
            round_num += 1
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=CLAUDE_TOOLS,
            )

            has_tool_use = False
            assistant_content: List[Any] = []
            tool_results_content: List[Dict[str, Any]] = []

            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    assistant_content.append(getattr(block, "text", "") or "")
                elif btype == "tool_use":
                    has_tool_use = True
                    tool_id = getattr(block, "id", "")
                    name = getattr(block, "name", "")
                    input_args = getattr(block, "input", {}) or {}
                    tool_calls_log.append({"tool": name, "input": input_args, "status": "running"})
                    result = execute_tool(name, **input_args)
                    if isinstance(result, dict) and "error" not in result:
                        structured_data[name] = result
                    result_str = json.dumps(result, default=str)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })
                    tool_calls_log[-1]["status"] = "complete"
                    tool_calls_log[-1]["result"] = result

            if not has_tool_use:
                reply = "".join(assistant_content) if assistant_content else ""
                return {
                    "reply": reply,
                    "tool_calls": tool_calls_log,
                    "structured_data": structured_data,
                }

            # Convert content blocks to API format (SDK may return objects)
            content_blocks = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    content_blocks.append({"type": "text", "text": getattr(block, "text", "") or ""})
                elif btype == "tool_use":
                    content_blocks.append({
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}) or {},
                    })
            messages.append({"role": "assistant", "content": content_blocks})
            messages.append({"role": "user", "content": tool_results_content})

        reply = "Maximum tool-use rounds reached."
        return {"reply": reply, "tool_calls": tool_calls_log, "structured_data": structured_data}

    except Exception as e:
        logger.exception("AI chat failed: %s", e)
        return {
            "reply": f"Error: {e}. Check ANTHROPIC_API_KEY and try again.",
            "tool_calls": tool_calls_log,
            "structured_data": structured_data,
        }


@router.post("/summarize")
async def summarize_text(body: SummarizeRequest):
    """
    Summarize article or text in 1-2 sentences. Used by News panel for AI-summarized articles.
    """
    if not body.text or not body.text.strip():
        return {"summary": "", "error": "Empty text"}
    try:
        if not ai_service.client:
            return {"summary": (body.text[:200] + "…") if len(body.text) > 200 else body.text, "error": "OpenAI not configured"}
        prompt = f"Summarize this financial news or text in 1-2 short sentences. Be factual and concise.\n\nText:\n{body.text[:3000]}"
        response = ai_service.client.chat.completions.create(
            model=ai_service.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        content = response.choices[0].message.content if response.choices else ""
        return {"summary": content.strip() if content else ""}
    except Exception as e:
        logger.warning("Summarize failed: %s", e)
        return {"summary": "", "error": str(e)}
