"""
Investor Report Generator

Generates professional, consumer-facing reports suitable for:
- Investor updates
- Fund fact sheets
- Performance summaries
- Research findings
- Strategy explanations

Uses OpenAI API to create polished, narrative-driven content.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics for reporting."""
    name: str
    type: str  # "Simple ML", "Ensemble", "LSTM"
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades_count: int
    best_trade: float
    worst_trade: float
    avg_trade_return: float
    profit_factor: float
    recovery_factor: float


@dataclass
class BacktestResults:
    """Complete backtest results for a strategy."""
    symbol: str
    period: str
    model_performance: ModelPerformance
    market_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    monthly_returns: Dict[str, float]


@dataclass
class InvestorReport:
    """Complete investor report structure."""
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


class InvestorReportGenerator:
    """Generate professional investor reports using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
        
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
    
    def _call_gpt(
        self,
        prompt: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Call OpenAI API to generate content.
        
        Args:
            prompt: Input prompt for GPT
            model: Model to use (gpt-4-turbo-preview, gpt-3.5-turbo, etc.)
            temperature: Creativity level (0-2, higher = more creative)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text from GPT
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional investment analyst and fund manager writing for institutional investors. Create polished, formal reports with insights and clear language."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def generate_executive_summary(
        self,
        models: List[ModelPerformance],
        period: str = "Q4 2024"
    ) -> str:
        """Generate executive summary of model performance."""
        models_text = "\n".join([
            f"- {m.name} ({m.type}): {m.annual_return:.1f}% return, {m.sharpe_ratio:.2f} Sharpe ratio"
            for m in models
        ])
        
        prompt = f"""
        Write a professional executive summary (300-400 words) for an investor report covering:
        
        Period: {period}
        
        Model Performance:
        {models_text}
        
        Create a compelling narrative that:
        1. Highlights key performance metrics
        2. Explains what drove outperformance
        3. Positions the strategies within market context
        4. Notes risk-adjusted returns
        
        Use professional language suitable for institutional investors.
        """
        
        return self._call_gpt(prompt, max_tokens=1500)
    
    def generate_strategy_overview(
        self,
        strategy_descriptions: Dict[str, str]
    ) -> str:
        """Generate overview of trading strategies."""
        strategies_text = "\n".join([
            f"- {name}: {desc}"
            for name, desc in strategy_descriptions.items()
        ])
        
        prompt = f"""
        Write a comprehensive strategy overview (400-500 words) for institutional investors explaining:
        
        Our Strategies:
        {strategies_text}
        
        Cover:
        1. Each strategy's core philosophy
        2. How signals are generated
        3. Risk controls and position sizing
        4. Expected behavior in different market environments
        5. Unique advantages vs. benchmarks
        
        Use professional, formal language.
        """
        
        return self._call_gpt(prompt, max_tokens=2000)
    
    def generate_research_findings(
        self,
        findings: List[Dict[str, str]],
        period: str = "Q4 2024"
    ) -> str:
        """Generate research findings section."""
        findings_text = "\n".join([
            f"- {f['title']}: {f['description']}"
            for f in findings
        ])
        
        prompt = f"""
        Write a research findings section (500-600 words) for an investor report covering:
        
        Period: {period}
        
        Key Findings:
        {findings_text}
        
        Create narrative that:
        1. Explains what we discovered about market dynamics
        2. Provides data-driven insights
        3. Discusses implications for future strategy
        4. Connects findings to model improvements
        5. Positions insights as proprietary research
        
        Use analytical, professional language.
        """
        
        return self._call_gpt(prompt, max_tokens=2500)
    
    def generate_performance_analysis(
        self,
        backtest_results: List[BacktestResults]
    ) -> str:
        """Generate detailed performance analysis."""
        results_text = "\n".join([
            f"- {r.symbol} ({r.period}): {r.model_performance.annual_return:.1f}% return, "
            f"{r.model_performance.sharpe_ratio:.2f} Sharpe, {r.model_performance.max_drawdown:.1f}% max DD"
            for r in backtest_results
        ])
        
        prompt = f"""
        Write a performance analysis section (600-800 words) analyzing:
        
        Backtest Results:
        {results_text}
        
        Include:
        1. Comparison to benchmarks (S&P 500, sector indices)
        2. Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
        3. Drawdown analysis and recovery periods
        4. Win rate and profit factor interpretation
        5. Monthly consistency and seasonal patterns
        6. Outperformance drivers by market regime
        7. Attribution analysis
        
        Use formal, institutional language with proper metric explanation.
        """
        
        return self._call_gpt(prompt, max_tokens=3000)
    
    def generate_risk_analysis(
        self,
        risk_metrics: Dict[str, float],
        var_metrics: Dict[str, float]
    ) -> str:
        """Generate risk analysis section."""
        risk_text = json.dumps(risk_metrics, indent=2)
        var_text = json.dumps(var_metrics, indent=2)
        
        prompt = f"""
        Write a comprehensive risk analysis section (400-500 words) for institutional investors covering:
        
        Risk Metrics:
        {risk_text}
        
        Value-at-Risk Metrics:
        {var_text}
        
        Address:
        1. Portfolio volatility and standard deviation
        2. Maximum drawdown in different market regimes
        3. Value-at-Risk (VaR) at 95% and 99% confidence levels
        4. Conditional Value-at-Risk (CVaR) implications
        5. Tail risk management strategies
        6. Stress testing results
        7. Correlation risk in multi-asset strategies
        8. Risk mitigation controls in place
        
        Use precise, professional risk terminology.
        """
        
        return self._call_gpt(prompt, max_tokens=2000)
    
    def generate_forward_looking_statements(
        self,
        market_outlook: str,
        strategy_adjustments: List[str]
    ) -> str:
        """Generate forward-looking statements section."""
        adjustments_text = "\n".join([f"- {adj}" for adj in strategy_adjustments])
        
        prompt = f"""
        Write a forward-looking statements section (300-400 words) suitable for institutional investors:
        
        Market Outlook:
        {market_outlook}
        
        Planned Strategy Adjustments:
        {adjustments_text}
        
        Include:
        1. Market environment expectations
        2. Anticipated risks and opportunities
        3. Strategy enhancements planned
        4. Investment thesis confidence
        5. Appropriate risk disclaimers
        
        Use professional forward-looking language with proper disclaimers.
        """
        
        return self._call_gpt(prompt, max_tokens=1500)
    
    def generate_full_report(
        self,
        title: str,
        models: List[ModelPerformance],
        backtest_results: List[BacktestResults],
        strategy_descriptions: Dict[str, str],
        research_findings: List[Dict[str, str]],
        risk_metrics: Dict[str, float],
        var_metrics: Dict[str, float],
        market_outlook: str,
        strategy_adjustments: List[str],
        period: str = "Q4 2024",
        fund_name: str = "Trading ML Fund"
    ) -> InvestorReport:
        """
        Generate complete investor report with all sections.
        
        Args:
            title: Report title
            models: List of ModelPerformance objects
            backtest_results: List of BacktestResults
            strategy_descriptions: Dict of strategy names to descriptions
            research_findings: List of research findings
            risk_metrics: Risk metrics dictionary
            var_metrics: VaR metrics dictionary
            market_outlook: Current market outlook
            strategy_adjustments: List of planned adjustments
            period: Reporting period
            fund_name: Fund/strategy name
            
        Returns:
            Complete InvestorReport object
        """
        logger.info(f"Generating investor report: {title}")
        
        # Generate each section
        executive_summary = self.generate_executive_summary(models, period)
        strategy_overview = self.generate_strategy_overview(strategy_descriptions)
        research_findings_text = self.generate_research_findings(research_findings, period)
        performance_analysis = self.generate_performance_analysis(backtest_results)
        risk_analysis = self.generate_risk_analysis(risk_metrics, var_metrics)
        forward_looking = self.generate_forward_looking_statements(market_outlook, strategy_adjustments)
        
        # Generate appendix with data tables
        appendix = self._generate_appendix(models, backtest_results)
        
        # Standard disclaimer
        disclaimer = self._generate_disclaimer(fund_name)
        
        return InvestorReport(
            title=title,
            date=datetime.now().strftime("%B %d, %Y"),
            executive_summary=executive_summary,
            strategy_overview=strategy_overview,
            research_findings=research_findings_text,
            performance_analysis=performance_analysis,
            risk_analysis=risk_analysis,
            forward_looking_statements=forward_looking,
            appendix=appendix,
            disclaimer=disclaimer
        )
    
    def _generate_appendix(
        self,
        models: List[ModelPerformance],
        backtest_results: List[BacktestResults]
    ) -> str:
        """Generate appendix with detailed data tables."""
        appendix = "## APPENDIX: DETAILED PERFORMANCE DATA\n\n"
        
        # Model performance table
        appendix += "### Model Performance Summary\n\n"
        appendix += "| Model | Type | Ann. Return | Sharpe | Max DD | Win Rate |\n"
        appendix += "|-------|------|------------|--------|--------|----------|\n"
        for m in models:
            appendix += f"| {m.name} | {m.type} | {m.annual_return:.1f}% | {m.sharpe_ratio:.2f} | {m.max_drawdown:.1f}% | {m.win_rate:.1f}% |\n"
        
        # Backtest results table
        appendix += "\n### Backtest Results by Symbol\n\n"
        appendix += "| Symbol | Period | Return | Sharpe | Max DD | Trades | Profit Factor |\n"
        appendix += "|--------|--------|--------|--------|--------|--------|---------------|\n"
        for r in backtest_results:
            mp = r.model_performance
            appendix += f"| {r.symbol} | {r.period} | {mp.annual_return:.1f}% | {mp.sharpe_ratio:.2f} | {mp.max_drawdown:.1f}% | {mp.trades_count} | {mp.profit_factor:.2f} |\n"
        
        return appendix
    
    def _generate_disclaimer(self, fund_name: str) -> str:
        """Generate standard legal disclaimer."""
        return f"""
## DISCLAIMER

This report is provided for informational purposes only and does not constitute an offer to buy or sell securities or a recommendation to buy or sell any security. The information contained in this report is based on sources believed to be reliable but is not guaranteed for accuracy or completeness.

Past performance is not indicative of future results. Investment in the {fund_name} strategies involves substantial risk of loss. Returns may be volatile and investors may lose their entire investment.

The strategies described herein may use leverage, short selling, derivatives, and other complex strategies which may not be suitable for all investors. Investors should consult with their financial advisors before investing.

Forward-looking statements contained in this report are subject to risks and uncertainties. Actual results may differ materially from those projected.

© 2024 {fund_name}. All rights reserved.
"""
    
    def export_report_to_markdown(
        self,
        report: InvestorReport,
        output_path: str
    ) -> str:
        """
        Export investor report to markdown file.
        
        Args:
            report: InvestorReport object
            output_path: Path to save markdown file
            
        Returns:
            Path to saved file
        """
        content = f"""# {report.title}

**Date**: {report.date}

---

## Executive Summary

{report.executive_summary}

---

## Strategy Overview

{report.strategy_overview}

---

## Research Findings

{report.research_findings}

---

## Performance Analysis

{report.performance_analysis}

---

## Risk Analysis

{report.risk_analysis}

---

## Forward-Looking Statements

{report.forward_looking_statements}

---

{report.appendix}

---

{report.disclaimer}
"""
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report exported to {output_path}")
        return output_path
    
    def export_report_to_html(
        self,
        report: InvestorReport,
        output_path: str
    ) -> str:
        """
        Export investor report to professional HTML.
        
        Args:
            report: InvestorReport object
            output_path: Path to save HTML file
            
        Returns:
            Path to saved file
        """
        # Convert markdown to HTML with professional styling
        try:
            import markdown
            md = markdown.Markdown(extensions=['tables', 'toc'])
        except ImportError:
            logger.warning("markdown package not installed. Using basic conversion.")
            md = None
        
        content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a3a52;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #0066cc;
            margin-top: 30px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .disclaimer {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 30px;
            font-size: 12px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 40px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report.title}</h1>
        <div class="meta">Report Date: {report.date}</div>
        
        <h2>Executive Summary</h2>
        <p>{report.executive_summary}</p>
        
        <hr>
        
        <h2>Strategy Overview</h2>
        <p>{report.strategy_overview}</p>
        
        <hr>
        
        <h2>Research Findings</h2>
        <p>{report.research_findings}</p>
        
        <hr>
        
        <h2>Performance Analysis</h2>
        <p>{report.performance_analysis}</p>
        
        <hr>
        
        <h2>Risk Analysis</h2>
        <p>{report.risk_analysis}</p>
        
        <hr>
        
        <h2>Forward-Looking Statements</h2>
        <p>{report.forward_looking_statements}</p>
        
        <hr>
        
        <h2>Appendix</h2>
        <pre>{report.appendix}</pre>
        
        <div class="disclaimer">
            <strong>Important Disclaimer:</strong><br>
            {report.disclaimer}
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report exported to {output_path}")
        return output_path
