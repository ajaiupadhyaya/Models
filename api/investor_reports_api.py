"""
Investor Report API Endpoints

Generate professional investor-facing reports with:
- Executive summaries
- Strategy analysis
- Performance metrics
- Risk analysis
- Research findings
- Forward-looking statements

All generated using OpenAI API for polished, narrative-driven content.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["investor-reports"])


# Request/Response Models
class ModelPerformanceData(BaseModel):
    """Model performance for report."""
    name: str
    type: str = Field(..., description="Simple ML, Ensemble, or LSTM")
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades_count: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_return: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0


class BacktestResultsData(BaseModel):
    """Backtest results for a symbol."""
    symbol: str
    period: str
    model_performance: ModelPerformanceData
    market_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    monthly_returns: Optional[Dict[str, float]] = None


class ResearchFinding(BaseModel):
    """Single research finding."""
    title: str
    description: str


class GenerateReportRequest(BaseModel):
    """Request to generate investor report."""
    title: str = Field(..., description="Report title")
    models: List[ModelPerformanceData]
    backtest_results: List[BacktestResultsData]
    strategy_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Dict of strategy names to descriptions"
    )
    research_findings: Optional[List[ResearchFinding]] = None
    risk_metrics: Optional[Dict[str, float]] = None
    var_metrics: Optional[Dict[str, float]] = None
    market_outlook: str = "Balanced with selective opportunities"
    strategy_adjustments: Optional[List[str]] = None
    period: str = "Q4 2024"
    fund_name: str = "Trading ML Fund"
    export_format: str = Field(default="markdown", description="markdown or html")


class ReportResponse(BaseModel):
    """Generated report response."""
    title: str
    date: str
    executive_summary: str
    strategy_overview: str
    research_findings: str
    performance_analysis: str
    risk_analysis: str
    forward_looking_statements: str
    appendix: str
    disclaimer: str
    file_path: Optional[str] = None


class QuickReportRequest(BaseModel):
    """Request for quick investor update."""
    title: str
    key_metrics: Dict[str, float]
    highlights: List[str]
    risks: List[str]
    next_steps: Optional[List[str]] = None


# Global report generator
_report_generator = None


def get_report_generator():
    """Get or initialize report generator."""
    global _report_generator
    if _report_generator is None:
        from core.investor_reports import InvestorReportGenerator
        _report_generator = InvestorReportGenerator()
    return _report_generator


@router.post("/generate")
async def generate_investor_report(request: GenerateReportRequest) -> ReportResponse:
    """
    Generate comprehensive investor report using OpenAI API.
    
    Creates professional, narrative-driven report suitable for:
    - Investor updates
    - Fund fact sheets
    - Quarterly reports
    - Performance summaries
    """
    try:
        generator = get_report_generator()
        if not generator.client:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API not configured. Set OPENAI_API_KEY environment variable."
            )
        
        # Convert request data to internal format
        from core.investor_reports import (
            ModelPerformance, BacktestResults, InvestorReportGenerator
        )
        
        models = [
            ModelPerformance(
                name=m.name,
                type=m.type,
                total_return=m.total_return,
                annual_return=m.annual_return,
                sharpe_ratio=m.sharpe_ratio,
                max_drawdown=m.max_drawdown,
                win_rate=m.win_rate,
                trades_count=m.trades_count,
                best_trade=m.best_trade,
                worst_trade=m.worst_trade,
                avg_trade_return=m.avg_trade_return,
                profit_factor=m.profit_factor,
                recovery_factor=m.recovery_factor
            )
            for m in request.models
        ]
        
        backtest_results = [
            BacktestResults(
                symbol=br.symbol,
                period=br.period,
                model_performance=ModelPerformance(
                    name=br.model_performance.name,
                    type=br.model_performance.type,
                    total_return=br.model_performance.total_return,
                    annual_return=br.model_performance.annual_return,
                    sharpe_ratio=br.model_performance.sharpe_ratio,
                    max_drawdown=br.model_performance.max_drawdown,
                    win_rate=br.model_performance.win_rate,
                    trades_count=br.model_performance.trades_count,
                    best_trade=br.model_performance.best_trade,
                    worst_trade=br.model_performance.worst_trade,
                    avg_trade_return=br.model_performance.avg_trade_return,
                    profit_factor=br.model_performance.profit_factor,
                    recovery_factor=br.model_performance.recovery_factor
                ),
                market_performance=br.market_performance,
                risk_metrics=br.risk_metrics,
                trading_metrics=br.trading_metrics,
                monthly_returns=br.monthly_returns or {}
            )
            for br in request.backtest_results
        ]
        
        findings = [
            {"title": f.title, "description": f.description}
            for f in (request.research_findings or [])
        ]
        
        # Generate report
        report = generator.generate_full_report(
            title=request.title,
            models=models,
            backtest_results=backtest_results,
            strategy_descriptions=request.strategy_descriptions,
            research_findings=findings,
            risk_metrics=request.risk_metrics or {},
            var_metrics=request.var_metrics or {},
            market_outlook=request.market_outlook,
            strategy_adjustments=request.strategy_adjustments or [],
            period=request.period,
            fund_name=request.fund_name
        )
        
        # Export to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investor_report_{timestamp}"
        
        if request.export_format == "html":
            file_path = f"data/reports/{filename}.html"
            generator.export_report_to_html(report, file_path)
        else:
            file_path = f"data/reports/{filename}.md"
            generator.export_report_to_markdown(report, file_path)
        
        return ReportResponse(
            title=report.title,
            date=report.date,
            executive_summary=report.executive_summary,
            strategy_overview=report.strategy_overview,
            research_findings=report.research_findings,
            performance_analysis=report.performance_analysis,
            risk_analysis=report.risk_analysis,
            forward_looking_statements=report.forward_looking_statements,
            appendix=report.appendix,
            disclaimer=report.disclaimer,
            file_path=file_path
        )
    
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-update")
async def generate_quick_update(request: QuickReportRequest) -> Dict[str, str]:
    """
    Generate quick investor update (not full report).
    
    For rapid updates on key metrics and highlights.
    """
    try:
        generator = get_report_generator()
        if not generator.client:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API not configured. Set OPENAI_API_KEY environment variable."
            )
        
        metrics_text = "\n".join([
            f"- {k}: {v}"
            for k, v in request.key_metrics.items()
        ])
        highlights_text = "\n".join([f"- {h}" for h in request.highlights])
        risks_text = "\n".join([f"- {r}" for r in request.risks])
        
        prompt = f"""
        Write a professional investor update (200-300 words) for: {request.title}
        
        Key Metrics:
        {metrics_text}
        
        Highlights:
        {highlights_text}
        
        Risks:
        {risks_text}
        
        Create a concise, compelling narrative suitable for quick distribution to investors.
        """
        
        update = generator._call_gpt(prompt, max_tokens=1000)
        
        return {
            "title": request.title,
            "date": datetime.now().isoformat(),
            "update": update,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to generate quick update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance-summary")
async def generate_performance_summary(
    symbol: str,
    period: str,
    annual_return: float,
    sharpe_ratio: float,
    max_drawdown: float
) -> Dict[str, str]:
    """
    Generate performance summary for a single strategy.
    """
    try:
        generator = get_report_generator()
        if not generator.client:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API not configured. Set OPENAI_API_KEY environment variable."
            )
        
        prompt = f"""
        Write a professional performance summary (150-200 words) for institutional investors:
        
        Strategy: {symbol}
        Period: {period}
        Annual Return: {annual_return:.1f}%
        Sharpe Ratio: {sharpe_ratio:.2f}
        Maximum Drawdown: {max_drawdown:.1f}%
        
        Create a compelling narrative that explains these metrics and their significance.
        Suitable for fact sheets and investor updates.
        """
        
        summary = generator._call_gpt(prompt, max_tokens=800)
        
        return {
            "symbol": symbol,
            "period": period,
            "summary": summary,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to generate performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples")
async def get_report_examples() -> Dict[str, str]:
    """Get example reports and templates."""
    return {
        "status": "success",
        "examples": {
            "executive_summary": "2-4 paragraphs on overall performance",
            "strategy_overview": "Explain core philosophy and mechanics",
            "research_findings": "Key discoveries and insights",
            "performance_analysis": "Detailed metrics and comparisons",
            "risk_analysis": "VaR, drawdown, stress test results",
            "forward_looking": "Outlook and planned changes"
        },
        "formats": ["markdown", "html"],
        "message": "POST to /api/v1/reports/generate with data"
    }


@router.post("/analyze-backtest")
async def analyze_backtest_results(results: BacktestResultsData) -> Dict[str, str]:
    """
    Analyze backtest results and generate narrative explanation.
    """
    try:
        generator = get_report_generator()
        if not generator.client:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API not configured. Set OPENAI_API_KEY environment variable."
            )
        
        mp = results.model_performance
        prompt = f"""
        Analyze these backtest results and explain what they mean for investors:
        
        Symbol: {results.symbol}
        Period: {results.period}
        
        Performance:
        - Annual Return: {mp.annual_return:.1f}%
        - Sharpe Ratio: {mp.sharpe_ratio:.2f}
        - Max Drawdown: {mp.max_drawdown:.1f}%
        - Win Rate: {mp.win_rate:.1f}%
        - Profit Factor: {mp.profit_factor:.2f}
        
        Trading:
        - Total Trades: {mp.trades_count}
        - Avg Trade Return: {mp.avg_trade_return:.2f}%
        - Best Trade: {mp.best_trade:.2f}%
        - Worst Trade: {mp.worst_trade:.2f}%
        
        Write a 300-400 word analysis explaining:
        1. What these metrics mean
        2. Strengths of the strategy
        3. Risk considerations
        4. Suitability for different investor types
        
        Use professional language.
        """
        
        analysis = generator._call_gpt(prompt, max_tokens=1500)
        
        return {
            "symbol": results.symbol,
            "period": results.period,
            "analysis": analysis,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to analyze backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def investor_reports_health() -> Dict[str, bool]:
    """Check investor report generator health."""
    try:
        generator = get_report_generator()
        return {
            "status": "healthy",
            "openai_configured": generator.client is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "openai_configured": False
        }
