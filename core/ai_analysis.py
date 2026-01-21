"""
AI Analysis Service

Uses OpenAI to provide:
- Chart/graph analysis and insights
- Market sentiment analysis
- Trading recommendations
- Plain-English data summaries
- Risk/opportunity analysis
"""

import os
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from openai import OpenAI, RateLimitError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class AIAnalysisService:
    """
    Automated AI analysis using OpenAI GPT.
    Analyzes charts, data, and market conditions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self.model = "gpt-4o-mini"  # Fast, capable model
        
        if not HAS_OPENAI:
            logger.warning("OpenAI package not installed; AI analysis disabled")
            return
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
    
    def analyze_price_chart(
        self,
        symbol: str,
        df: pd.DataFrame,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Analyze price chart and return plain-English summary.
        
        Args:
            symbol: Stock ticker
            df: OHLCV DataFrame
            metrics: Optional dict of technical metrics (RSI, MACD, etc.)
        
        Returns:
            Plain-English analysis
        """
        if not self.client:
            return "AI analysis unavailable (OpenAI not configured)"
        
        try:
            # Extract key price metrics
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-5] if len(df) > 5 else df['Close'].iloc[0]
            change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price else 0
            
            high_52w = df['Close'].max() if len(df) >= 252 else df['Close'].max()
            low_52w = df['Close'].min() if len(df) >= 252 else df['Close'].min()
            
            vol_avg = df['Volume'].mean()
            vol_current = df['Volume'].iloc[-1]
            
            # Build analysis context
            context = f"""
            Analyze the following stock data for {symbol}:
            - Current Price: ${current_price:.2f}
            - 5-Day Change: {change_pct:.2f}%
            - 52-Week High/Low: ${high_52w:.2f} / ${low_52w:.2f}
            - Current Volume: {vol_current:,.0f} (Avg: {vol_avg:,.0f})
            - Days of Data: {len(df)}
            """
            
            if metrics:
                context += "\nTechnical Metrics:\n"
                for k, v in metrics.items():
                    context += f"  - {k}: {v}\n"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Provide concise, actionable insights about stock price charts."
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nProvide a brief 2-3 sentence analysis of the price action, trend, and outlook."
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except RateLimitError:
            logger.warning("OpenAI rate limit hit; skipping analysis")
            return "Analysis temporarily unavailable (rate limit)"
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return f"Analysis error: {str(e)[:100]}"
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of market news or analysis text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment score, label, and explanation
        """
        if not self.client:
            return {"sentiment": "neutral", "score": 0.0, "error": "AI disabled"}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze market sentiment. Respond with JSON: {\"sentiment\": \"bullish|bearish|neutral\", \"score\": -1.0 to 1.0, \"reasoning\": \"...\"}"
                    },
                    {
                        "role": "user",
                        "content": f"Analyze sentiment: {text}"
                    }
                ],
                max_tokens=150,
                temperature=0.5
            )
            
            content = response.choices[0].message.content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"sentiment": "neutral", "score": 0.0, "reasoning": content}
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment": "error", "score": 0.0, "error": str(e)}
    
    def generate_trading_insight(
        self,
        symbol: str,
        current_price: float,
        prediction: float,
        confidence: float,
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered trading insight and recommendation.
        
        Args:
            symbol: Stock ticker
            current_price: Current stock price
            prediction: ML model's price prediction (next period)
            confidence: Model confidence (0-1)
            market_context: Optional market context
        
        Returns:
            Dict with recommendation, reasoning, risk
        """
        if not self.client:
            return {"recommendation": "HOLD", "reasoning": "AI disabled"}
        
        try:
            implied_move = ((prediction - current_price) / current_price * 100)
            
            prompt = f"""
            Stock: {symbol}
            Current Price: ${current_price:.2f}
            ML Prediction: ${prediction:.2f} (implied move: {implied_move:+.2f}%)
            Model Confidence: {confidence*100:.0f}%
            """
            
            if market_context:
                prompt += f"\nMarket Context: {market_context}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional trader. Provide trading recommendations with clear reasoning. Respond as JSON: {\"action\": \"BUY|SELL|HOLD\", \"reasoning\": \"...\", \"risk_level\": \"low|medium|high\", \"stop_loss_pct\": X, \"take_profit_pct\": Y}"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=250,
                temperature=0.6
            )
            
            content = response.choices[0].message.content.strip()
            try:
                result = json.loads(content)
                result['timestamp'] = datetime.now().isoformat()
                return result
            except json.JSONDecodeError:
                return {
                    "action": "HOLD",
                    "reasoning": content,
                    "risk_level": "medium",
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Trading insight error: {e}")
            return {"action": "HOLD", "reasoning": f"Error: {str(e)[:100]}", "risk_level": "unknown"}
    
    def explain_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Explain financial metrics in plain English.
        
        Args:
            metrics: Dict of metrics (Sharpe, Sortino, Max Drawdown, etc.)
        
        Returns:
            Plain-English explanation
        """
        if not self.client:
            return "Explanation unavailable (AI disabled)"
        
        try:
            metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Explain financial metrics in simple, plain English for non-experts."
                    },
                    {
                        "role": "user",
                        "content": f"Explain what these metrics mean:\n{metrics_text}"
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Metric explanation error: {e}")
            return f"Explanation unavailable: {str(e)[:50]}"


# Global instance
_service = None


def get_ai_service() -> AIAnalysisService:
    """Get or create the global AI service."""
    global _service
    if _service is None:
        _service = AIAnalysisService()
    return _service
