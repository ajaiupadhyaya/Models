#!/usr/bin/env python3
"""
Quick Start: Generate Your First Investor Report

This script demonstrates the complete workflow for generating
a professional investor report with sample data.

Run from project root:
    python automation/quick_investor_report.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.investor_reports import (
    InvestorReportGenerator,
    ModelPerformance,
    BacktestResults,
)
from datetime import datetime


def create_sample_models():
    """Create sample trading model performance data."""
    return [
        ModelPerformance(
            name="Ensemble ML Strategy",
            type="Ensemble",
            total_return=45.2,
            annual_return=15.1,
            sharpe_ratio=1.85,
            max_drawdown=-8.3,
            win_rate=58.2,
            trades_count=342,
            best_trade=8.5,
            worst_trade=-4.2,
            avg_trade_return=0.132,
            profit_factor=2.15,
            recovery_factor=5.44
        ),
        ModelPerformance(
            name="LSTM Deep Learning",
            type="LSTM",
            total_return=38.7,
            annual_return=12.9,
            sharpe_ratio=1.62,
            max_drawdown=-11.2,
            win_rate=52.5,
            trades_count=285,
            best_trade=7.8,
            worst_trade=-5.1,
            avg_trade_return=0.136,
            profit_factor=1.88,
            recovery_factor=3.45
        ),
        ModelPerformance(
            name="Statistical Arbitrage",
            type="Statistical",
            total_return=52.1,
            annual_return=17.4,
            sharpe_ratio=2.12,
            max_drawdown=-6.7,
            win_rate=61.2,
            trades_count=421,
            best_trade=9.2,
            worst_trade=-3.8,
            avg_trade_return=0.124,
            profit_factor=2.42,
            recovery_factor=7.79
        )
    ]


def create_sample_backtest_results(models):
    """Create sample backtest results for different symbols."""
    return [
        BacktestResults(
            symbol="SPY",
            period="2023-2024",
            model_performance=models[0],
            market_performance={
                "annual_return": 24.9,
                "sharpe_ratio": 1.95,
                "max_drawdown": -12.3
            },
            risk_metrics={
                "var_95": -2.1,
                "var_99": -3.5,
                "cvar_95": -2.8,
                "correlation_market": 0.45
            },
            trading_metrics={
                "avg_trade_duration_days": 8.2,
                "consecutive_wins": 12,
                "consecutive_losses": 5,
                "largest_win_streak_return": 18.5
            },
            monthly_returns={
                "2024-01": 3.2,
                "2024-02": 1.8,
                "2024-03": 2.5,
                "2024-04": 4.1,
                "2024-05": 0.9,
                "2024-06": 2.3
            }
        ),
        BacktestResults(
            symbol="QQQ",
            period="2023-2024",
            model_performance=models[1],
            market_performance={
                "annual_return": 36.2,
                "sharpe_ratio": 2.15,
                "max_drawdown": -15.7
            },
            risk_metrics={
                "var_95": -2.8,
                "var_99": -4.2,
                "cvar_95": -3.5,
                "correlation_market": 0.62
            },
            trading_metrics={
                "avg_trade_duration_days": 6.5,
                "consecutive_wins": 9,
                "consecutive_losses": 6,
                "largest_win_streak_return": 16.2
            }
        ),
        BacktestResults(
            symbol="IWM",
            period="2023-2024",
            model_performance=models[2],
            market_performance={
                "annual_return": 18.5,
                "sharpe_ratio": 1.45,
                "max_drawdown": -14.2
            },
            risk_metrics={
                "var_95": -2.3,
                "var_99": -3.6,
                "cvar_95": -3.0,
                "correlation_market": 0.75
            },
            trading_metrics={
                "avg_trade_duration_days": 7.8,
                "consecutive_wins": 14,
                "consecutive_losses": 4,
                "largest_win_streak_return": 22.1
            }
        )
    ]


def main():
    """Generate sample investor report."""
    print("\n" + "="*80)
    print("INVESTOR REPORT GENERATION - QUICK START")
    print("="*80 + "\n")

    # Step 1: Initialize generator
    print("📊 Initializing report generator...")
    generator = InvestorReportGenerator()

    if not generator.client:
        print("\n⚠️  OpenAI API Key Not Configured")
        print("\nTo use this feature, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nGet your API key at: https://platform.openai.com/api-keys")
        print("\n💡 Note: You can still view the example code and structure below.\n")
        show_example_code()
        return

    print("✓ Generator ready (OpenAI API configured)\n")

    # Step 2: Prepare data
    print("📈 Creating sample model data...")
    models = create_sample_models()
    print(f"✓ Created {len(models)} trading models")

    print("\n📉 Creating sample backtest results...")
    backtest_results = create_sample_backtest_results(models)
    print(f"✓ Created {len(backtest_results)} backtest scenarios")

    # Step 3: Define research findings
    print("\n🔬 Adding research findings...")
    research_findings = [
        {
            "title": "Volatility Mean Reversion Signal",
            "description": "Ensemble model identified robust mean-reversion pattern in VIX futures"
        },
        {
            "title": "Cross-Asset Correlation Decay",
            "description": "Long-term equity-bond correlations declining, presenting diversification opportunities"
        },
        {
            "title": "Earnings Surprise Alpha",
            "description": "ML models detect pre-earnings drift patterns exceeding traditional indicators by 2.3x"
        },
        {
            "title": "Macro Factor Loading Changes",
            "description": "Interest rate sensitivity in equities shifted from negative to neutral"
        }
    ]
    print(f"✓ Added {len(research_findings)} research findings")

    # Step 4: Generate report
    print("\n🚀 Generating investor report... (this may take 1-2 minutes)\n")
    print("-" * 80)

    try:
        report = generator.generate_full_report(
            title="Q2 2024 Investment Update",
            models=models,
            backtest_results=backtest_results,
            strategy_descriptions={
                "Ensemble ML Strategy": "Multi-model ensemble combining statistical, ML, and RL approaches",
                "LSTM Deep Learning": "Sequential pattern recognition using long short-term memory networks",
                "Statistical Arbitrage": "Pairs trading and mean-reversion based on cointegration analysis"
            },
            research_findings=research_findings,
            risk_metrics={
                "portfolio_var_95": -2.4,
                "portfolio_var_99": -3.8,
                "stress_test_2008": -18.5,
                "stress_test_covid": -14.2,
                "liquidity_coverage_ratio": 2.8
            },
            var_metrics={
                "daily_var_95": -2.1,
                "daily_var_99": -3.2,
                "cvar_95": -2.9,
                "expected_shortfall": -3.1
            },
            market_outlook="Balanced positioning with selective opportunities in growth equities",
            strategy_adjustments=[
                "Increased allocation to volatility-harvesting strategies",
                "Enhanced risk management for macro tail events",
                "Expanded Asia-Pacific opportunity set"
            ],
            period="Q2 2024",
            fund_name="Trading ML Fund"
        )

        print("-" * 80)
        print("\n✅ Report generated successfully!\n")

        # Step 5: Export to files
        print("💾 Exporting report to files...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create reports directory if needed
        (project_root / "data" / "reports").mkdir(parents=True, exist_ok=True)

        # Export to markdown
        md_path = project_root / f"data/reports/investor_report_{timestamp}.md"
        generator.export_report_to_markdown(report, str(md_path))
        print(f"✓ Markdown: {md_path}")

        # Export to HTML
        html_path = project_root / f"data/reports/investor_report_{timestamp}.html"
        generator.export_report_to_html(report, str(html_path))
        print(f"✓ HTML: {html_path}")

        # Step 6: Display summary
        print("\n" + "="*80)
        print("REPORT SUMMARY")
        print("="*80)
        print(f"\nTitle: {report.title}")
        print(f"Date: {report.date}")
        print(f"\nSection Lengths:")
        print(f"  Executive Summary: {len(report.executive_summary)} chars")
        print(f"  Strategy Overview: {len(report.strategy_overview)} chars")
        print(f"  Research Findings: {len(report.research_findings)} chars")
        print(f"  Performance Analysis: {len(report.performance_analysis)} chars")
        print(f"  Risk Analysis: {len(report.risk_analysis)} chars")
        print(f"  Forward-Looking: {len(report.forward_looking_statements)} chars")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. View the reports:")
        print(f"   • Markdown: open {md_path}")
        print(f"   • HTML: open {html_path}")

        print("\n2. Integrate with your data:")
        print("   • Use real backtest results from your models")
        print("   • Include actual research findings and insights")
        print("   • Customize for your fund/strategy")

        print("\n3. API Integration:")
        print("   • Start API: uvicorn api.main:app --port 8000")
        print("   • POST to /api/v1/reports/generate")
        print("   • See API_DOCUMENTATION.md for API details")

        print("\n4. Automation:")
        print("   • Schedule weekly/monthly report generation")
        print("   • Email reports to investors")
        print("   • Archive reports for compliance")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        print("\nTroubleshooting:")
        print("1. Check OpenAI API key is valid")
        print("2. Verify API account has credits")
        print("3. Check internet connection")
        print("4. Review error message above")


def show_example_code():
    """Show example code structure."""
    example = '''
EXAMPLE CODE STRUCTURE
======================

from core.investor_reports import InvestorReportGenerator, ModelPerformance, BacktestResults

# Initialize
generator = InvestorReportGenerator()

# Create model data
models = [
    ModelPerformance(
        name="Strategy Name",
        type="Ensemble",
        total_return=45.2,
        annual_return=15.1,
        sharpe_ratio=1.85,
        max_drawdown=-8.3,
        win_rate=58.2,
        # ... more metrics
    )
]

# Create backtest results
backtest_results = [
    BacktestResults(
        symbol="SPY",
        period="2023-2024",
        model_performance=models[0],
        market_performance={...},
        risk_metrics={...},
        trading_metrics={...}
    )
]

# Generate report
report = generator.generate_full_report(
    title="Q2 2024 Update",
    models=models,
    backtest_results=backtest_results,
    # ... additional parameters
)

# Export
generator.export_report_to_markdown(report, "report.md")
generator.export_report_to_html(report, "report.html")
    '''
    print(example)


if __name__ == "__main__":
    main()
