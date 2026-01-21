"""
Company Analysis API

Complete automated analysis for any company with comprehensive reports.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.company_search import CompanySearch, search_companies
from models.fundamental.company_analyzer import CompanyAnalyzer
from models.valuation.dcf_model import DCFModel
from models.risk.var_cvar import VaRModel, CVaRModel
from core.data_fetcher import DataFetcher
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class CompanySearchRequest(BaseModel):
    """Company search request."""
    query: str = Field(..., description="Search query (company name or ticker)")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    min_score: int = Field(60, ge=0, le=100, description="Minimum match score")


class CompanySearchResponse(BaseModel):
    """Company search response."""
    results: List[Dict[str, Any]]
    count: int
    query: str


class CompanyAnalysisRequest(BaseModel):
    """Company analysis request."""
    ticker: str = Field(..., description="Stock ticker symbol")
    include_dcf: bool = Field(True, description="Include DCF valuation")
    include_risk: bool = Field(True, description="Include risk analysis")
    include_technicals: bool = Field(True, description="Include technical analysis")
    period: str = Field("1y", description="Historical data period")


class CompanyAnalysisResponse(BaseModel):
    """Company analysis response."""
    ticker: str
    company_name: str
    analysis_date: str
    fundamental_analysis: Dict[str, Any]
    valuation: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any]


# API Endpoints
@router.get("/search", response_model=CompanySearchResponse)
async def search_company(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    min_score: int = Query(60, ge=0, le=100)
):
    """
    Search for companies by name or ticker.
    
    Supports fuzzy matching for company names and ticker symbols.
    """
    try:
        searcher = CompanySearch()
        results = searcher.search(query, limit=limit, min_score=min_score)
        
        return CompanySearchResponse(
            results=results,
            count=len(results),
            query=query
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/{ticker}")
async def validate_ticker(ticker: str):
    """
    Validate if a ticker exists and is tradeable.
    """
    try:
        searcher = CompanySearch()
        is_valid, message = searcher.validate_ticker(ticker)
        
        return {
            "ticker": ticker.upper(),
            "valid": is_valid,
            "message": message
        }
    
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{ticker}", response_model=CompanyAnalysisResponse)
async def analyze_company(
    ticker: str,
    include_dcf: bool = Query(True),
    include_risk: bool = Query(True),
    include_technicals: bool = Query(True),
    period: str = Query("1y")
):
    """
    Complete automated analysis of a company.
    
    Provides:
    - Fundamental analysis (financials, ratios, metrics)
    - DCF valuation (optional)
    - Risk metrics (VaR, CVaR, beta, volatility)
    - Technical analysis (trends, signals)
    - Summary and recommendations
    """
    try:
        # Validate ticker
        searcher = CompanySearch()
        is_valid, message = searcher.validate_ticker(ticker)
        if not is_valid:
            raise HTTPException(status_code=404, detail=message)
        
        # Initialize analyzer
        analyzer = CompanyAnalyzer(ticker)
        
        # Get comprehensive analysis
        logger.info(f"Analyzing {ticker}...")
        fundamental_analysis = analyzer.comprehensive_analysis()
        
        response = {
            "ticker": ticker.upper(),
            "company_name": fundamental_analysis['profile']['name'],
            "analysis_date": datetime.now().isoformat(),
            "fundamental_analysis": fundamental_analysis,
            "summary": {}
        }
        
        # DCF Valuation
        if include_dcf:
            try:
                logger.info(f"Calculating DCF for {ticker}...")
                dcf_result = _calculate_dcf(analyzer)
                response["valuation"] = dcf_result
            except Exception as e:
                logger.warning(f"DCF calculation failed: {e}")
                response["valuation"] = {"error": str(e)}
        
        # Risk Analysis
        if include_risk:
            try:
                logger.info(f"Calculating risk metrics for {ticker}...")
                risk_metrics = _calculate_risk_metrics(ticker, period)
                response["risk_metrics"] = risk_metrics
            except Exception as e:
                logger.warning(f"Risk calculation failed: {e}")
                response["risk_metrics"] = {"error": str(e)}
        
        # Technical Analysis
        if include_technicals:
            try:
                logger.info(f"Performing technical analysis for {ticker}...")
                technical = _calculate_technical_analysis(ticker, period)
                response["technical_analysis"] = technical
            except Exception as e:
                logger.warning(f"Technical analysis failed: {e}")
                response["technical_analysis"] = {"error": str(e)}
        
        # Generate Summary
        response["summary"] = _generate_summary(response)
        
        logger.info(f"Analysis complete for {ticker}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sectors")
async def get_sectors():
    """Get list of available sectors."""
    sectors = [
        "Technology",
        "Healthcare",
        "Financial Services",
        "Consumer Cyclical",
        "Industrials",
        "Communication Services",
        "Consumer Defensive",
        "Energy",
        "Real Estate",
        "Basic Materials",
        "Utilities"
    ]
    return {"sectors": sectors}


@router.get("/sector/{sector}")
async def get_sector_companies(
    sector: str,
    limit: int = Query(50, ge=1, le=200)
):
    """Get companies in a specific sector."""
    try:
        searcher = CompanySearch()
        companies = searcher.filter_by_sector(sector)
        
        return {
            "sector": sector,
            "companies": companies[:limit],
            "count": len(companies)
        }
    
    except Exception as e:
        logger.error(f"Sector query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-companies")
async def get_top_companies(n: int = Query(50, ge=1, le=200)):
    """Get top N companies by market cap."""
    try:
        searcher = CompanySearch()
        companies = searcher.get_top_companies(n)
        
        return {
            "companies": companies,
            "count": len(companies)
        }
    
    except Exception as e:
        logger.error(f"Top companies error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Helper Functions
def _calculate_dcf(analyzer: CompanyAnalyzer) -> Dict[str, Any]:
    """Calculate DCF valuation."""
    try:
        # Get financial data
        financials = analyzer.financials
        cash_flow = analyzer.cash_flow
        info = analyzer.info
        
        if len(cash_flow.columns) == 0:
            return {"error": "Insufficient cash flow data"}
        
        # Get free cash flow
        latest_cf = cash_flow.iloc[:, 0]
        fcf = latest_cf.get('Free Cash Flow', 0)
        
        if fcf <= 0:
            return {"error": "No positive free cash flow"}
        
        # DCF parameters
        growth_rate = info.get('revenueGrowth', 0.05)
        terminal_growth = 0.025
        discount_rate = 0.10
        shares_outstanding = info.get('sharesOutstanding', 1)
        
        # Calculate DCF
        dcf = DCFModel(
            free_cash_flows=[fcf],
            growth_rate=growth_rate,
            terminal_growth_rate=terminal_growth,
            discount_rate=discount_rate,
            years_to_project=5
        )
        
        intrinsic_value = dcf.calculate_enterprise_value()
        value_per_share = intrinsic_value / shares_outstanding if shares_outstanding > 0 else 0
        current_price = info.get('currentPrice', 0)
        
        return {
            "intrinsic_value": intrinsic_value,
            "value_per_share": value_per_share,
            "current_price": current_price,
            "upside": ((value_per_share - current_price) / current_price * 100) if current_price > 0 else 0,
            "parameters": {
                "fcf": fcf,
                "growth_rate": growth_rate,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate,
                "shares_outstanding": shares_outstanding
            }
        }
    
    except Exception as e:
        return {"error": str(e)}


def _calculate_risk_metrics(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Calculate risk metrics."""
    try:
        fetcher = DataFetcher()
        data = fetcher.get_stock_data(ticker, period=period)
        
        if len(data) == 0:
            return {"error": "No price data available"}
        
        returns = data['Close'].pct_change().dropna()
        
        # Calculate VaR and CVaR
        var_95 = VaRModel.calculate_var(returns, confidence_level=0.95)
        var_99 = VaRModel.calculate_var(returns, confidence_level=0.99)
        cvar_95 = CVaRModel.calculate_cvar(returns, confidence_level=0.95)
        cvar_99 = CVaRModel.calculate_cvar(returns, confidence_level=0.99)
        
        # Volatility
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252)
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        rf_rate = 0.02
        sharpe = (returns.mean() * 252 - rf_rate) / volatility_annual if volatility_annual > 0 else 0
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "volatility_daily": volatility_daily,
            "volatility_annual": volatility_annual,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "returns_mean_annual": returns.mean() * 252,
            "returns_std_annual": volatility_annual
        }
    
    except Exception as e:
        return {"error": str(e)}


def _calculate_technical_analysis(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Calculate technical indicators."""
    try:
        fetcher = DataFetcher()
        data = fetcher.get_stock_data(ticker, period=period)
        
        if len(data) < 50:
            return {"error": "Insufficient data for technical analysis"}
        
        close = data['Close']
        
        # Moving averages
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        
        current_price = close.iloc[-1]
        
        # Trend
        trend = "bullish" if ma_20 > ma_50 else "bearish"
        if ma_200:
            if current_price > ma_200:
                long_term_trend = "bullish"
            else:
                long_term_trend = "bearish"
        else:
            long_term_trend = "insufficient_data"
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Signals
        signals = []
        if current_rsi < 30:
            signals.append("oversold")
        elif current_rsi > 70:
            signals.append("overbought")
        
        if ma_20 > ma_50 and trend == "bullish":
            signals.append("golden_cross_short_term")
        elif ma_20 < ma_50 and trend == "bearish":
            signals.append("death_cross_short_term")
        
        return {
            "current_price": current_price,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "rsi": current_rsi,
            "trend_short_term": trend,
            "trend_long_term": long_term_trend,
            "signals": signals,
            "price_vs_ma20": ((current_price - ma_20) / ma_20 * 100) if ma_20 else 0,
            "price_vs_ma50": ((current_price - ma_50) / ma_50 * 100) if ma_50 else 0,
        }
    
    except Exception as e:
        return {"error": str(e)}


def _generate_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate analysis summary with key insights."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "ticker": analysis["ticker"],
        "company": analysis["company_name"]
    }
    
    # Key metrics
    fundamental = analysis.get("fundamental_analysis", {})
    
    # Valuation summary
    valuation = fundamental.get("valuation", {})
    summary["valuation_grade"] = _grade_valuation(valuation)
    
    # Profitability summary
    profitability = fundamental.get("profitability", {})
    summary["profitability_grade"] = _grade_profitability(profitability)
    
    # Financial health summary
    health = fundamental.get("financial_health", {})
    summary["financial_health_grade"] = _grade_financial_health(health)
    
    # Growth summary
    growth = fundamental.get("growth", {})
    summary["growth_grade"] = _grade_growth(growth)
    
    # Overall score
    grades = [
        summary["valuation_grade"]["score"],
        summary["profitability_grade"]["score"],
        summary["financial_health_grade"]["score"],
        summary["growth_grade"]["score"]
    ]
    summary["overall_score"] = np.mean([g for g in grades if g is not None])
    summary["overall_grade"] = _score_to_letter(summary["overall_score"])
    
    # Investment recommendation
    summary["recommendation"] = _generate_recommendation(summary, analysis)
    
    return summary


def _grade_valuation(valuation: Dict) -> Dict:
    """Grade valuation metrics."""
    score = 50  # Start neutral
    
    pe = valuation.get("pe_ratio")
    pb = valuation.get("price_to_book")
    
    if pe:
        if pe < 15:
            score += 20
        elif pe < 25:
            score += 10
        elif pe > 40:
            score -= 20
    
    if pb:
        if pb < 1:
            score += 15
        elif pb < 3:
            score += 5
        elif pb > 5:
            score -= 15
    
    return {
        "score": max(0, min(100, score)),
        "letter": _score_to_letter(score),
        "summary": "Attractive" if score > 70 else "Fair" if score > 50 else "Expensive"
    }


def _grade_profitability(profitability: Dict) -> Dict:
    """Grade profitability metrics."""
    score = 50
    
    net_margin = profitability.get("net_profit_margin")
    roe = profitability.get("roe")
    
    if net_margin:
        if net_margin > 0.20:
            score += 25
        elif net_margin > 0.10:
            score += 15
        elif net_margin < 0:
            score -= 30
    
    if roe:
        if roe > 0.20:
            score += 25
        elif roe > 0.10:
            score += 15
        elif roe < 0:
            score -= 30
    
    return {
        "score": max(0, min(100, score)),
        "letter": _score_to_letter(score),
        "summary": "Excellent" if score > 70 else "Good" if score > 50 else "Poor"
    }


def _grade_financial_health(health: Dict) -> Dict:
    """Grade financial health."""
    score = 50
    
    current_ratio = health.get("current_ratio")
    debt_to_equity = health.get("debt_to_equity")
    
    if current_ratio:
        if current_ratio > 2:
            score += 20
        elif current_ratio > 1:
            score += 10
        elif current_ratio < 1:
            score -= 20
    
    if debt_to_equity:
        if debt_to_equity < 0.5:
            score += 20
        elif debt_to_equity < 1:
            score += 10
        elif debt_to_equity > 2:
            score -= 20
    
    return {
        "score": max(0, min(100, score)),
        "letter": _score_to_letter(score),
        "summary": "Strong" if score > 70 else "Adequate" if score > 50 else "Weak"
    }


def _grade_growth(growth: Dict) -> Dict:
    """Grade growth metrics."""
    score = 50
    
    revenue_growth = growth.get("revenue_growth_yoy")
    earnings_growth = growth.get("earnings_growth_yoy")
    
    if revenue_growth:
        if revenue_growth > 0.20:
            score += 25
        elif revenue_growth > 0.10:
            score += 15
        elif revenue_growth < 0:
            score -= 20
    
    if earnings_growth:
        if earnings_growth > 0.20:
            score += 25
        elif earnings_growth > 0.10:
            score += 15
        elif earnings_growth < 0:
            score -= 20
    
    return {
        "score": max(0, min(100, score)),
        "letter": _score_to_letter(score),
        "summary": "High Growth" if score > 70 else "Moderate Growth" if score > 50 else "Low/Negative Growth"
    }


def _score_to_letter(score: float) -> str:
    """Convert numerical score to letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 80:
        return "A-"
    elif score >= 75:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 65:
        return "B-"
    elif score >= 60:
        return "C+"
    elif score >= 55:
        return "C"
    elif score >= 50:
        return "C-"
    elif score >= 40:
        return "D"
    else:
        return "F"


def _generate_recommendation(summary: Dict, analysis: Dict) -> Dict:
    """Generate investment recommendation."""
    score = summary["overall_score"]
    
    if score >= 75:
        rating = "Strong Buy"
    elif score >= 65:
        rating = "Buy"
    elif score >= 55:
        rating = "Hold"
    elif score >= 45:
        rating = "Underperform"
    else:
        rating = "Sell"
    
    # Key points
    points = []
    
    if summary["profitability_grade"]["score"] > 70:
        points.append("Strong profitability metrics")
    if summary["growth_grade"]["score"] > 70:
        points.append("Impressive growth trajectory")
    if summary["financial_health_grade"]["score"] > 70:
        points.append("Solid financial position")
    if summary["valuation_grade"]["score"] > 70:
        points.append("Attractive valuation")
    
    # Risks
    risks = []
    if summary["financial_health_grade"]["score"] < 50:
        risks.append("Weak balance sheet")
    if summary["growth_grade"]["score"] < 50:
        risks.append("Declining growth")
    if summary["valuation_grade"]["score"] < 50:
        risks.append("High valuation")
    
    return {
        "rating": rating,
        "score": score,
        "key_strengths": points,
        "key_risks": risks,
        "confidence": "High" if len(points) >= 3 else "Medium" if len(points) >= 2 else "Low"
    }
