#!/usr/bin/env python3
"""
Company Analyzer CLI

Interactive command-line tool for company search and analysis.

Usage:
    python analyze_company.py
    python analyze_company.py TSLA
    python analyze_company.py --search "Tesla"
    python analyze_company.py AAPL --full --export report.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.company_search import CompanySearch
from models.fundamental.company_analyzer import CompanyAnalyzer
from api.company_analysis_api import (
    _calculate_dcf,
    _calculate_risk_metrics,
    _calculate_technical_analysis,
    _generate_summary
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_section(title: str):
    """Print section title."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'─' * 80}{Colors.END}")


def print_metric(label: str, value: Any, color: str = Colors.END):
    """Print formatted metric."""
    print(f"  {Colors.BOLD}{label}:{Colors.END} {color}{value}{Colors.END}")


def format_currency(value: float) -> str:
    """Format value as currency."""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value*100:.2f}%" if value is not None else "N/A"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with decimals."""
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def search_interactive():
    """Interactive company search."""
    print_header("COMPANY SEARCH")
    
    searcher = CompanySearch()
    
    query = input(f"{Colors.BOLD}Enter company name or ticker: {Colors.END}").strip()
    if not query:
        print(f"{Colors.RED}Search query cannot be empty{Colors.END}")
        return None
    
    print(f"\n{Colors.CYAN}Searching for '{query}'...{Colors.END}\n")
    results = searcher.search(query, limit=10)
    
    if not results:
        print(f"{Colors.RED}No companies found matching '{query}'{Colors.END}")
        return None
    
    print(f"{Colors.GREEN}Found {len(results)} matches:{Colors.END}\n")
    
    for i, company in enumerate(results, 1):
        ticker = company.get('ticker', 'N/A')
        name = company.get('name', 'N/A')
        sector = company.get('sector', 'N/A')
        market_cap = company.get('market_cap', 0)
        score = company.get('match_score', 0)
        
        print(f"{Colors.BOLD}[{i}]{Colors.END} {Colors.GREEN}{ticker}{Colors.END} - {name}")
        print(f"    Sector: {sector} | Market Cap: {format_currency(market_cap)} | Match: {score}%")
    
    # Let user select
    while True:
        selection = input(f"\n{Colors.BOLD}Select company (1-{len(results)}) or 'q' to quit: {Colors.END}").strip()
        
        if selection.lower() == 'q':
            return None
        
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(results):
                return results[idx]['ticker']
            else:
                print(f"{Colors.RED}Invalid selection. Please choose 1-{len(results)}{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Invalid input. Please enter a number or 'q'{Colors.END}")


def analyze_company(ticker: str, full_analysis: bool = True, export_path: Optional[str] = None):
    """Analyze a company and display results."""
    print_header(f"ANALYZING {ticker.upper()}")
    
    # Validate ticker
    searcher = CompanySearch()
    is_valid, message = searcher.validate_ticker(ticker)
    
    if not is_valid:
        print(f"{Colors.RED}Error: {message}{Colors.END}")
        return
    
    print(f"{Colors.GREEN}✓ {message}{Colors.END}\n")
    
    # Initialize analyzer
    try:
        analyzer = CompanyAnalyzer(ticker)
        analysis = analyzer.comprehensive_analysis()
    except Exception as e:
        print(f"{Colors.RED}Error analyzing {ticker}: {e}{Colors.END}")
        return
    
    # Display results
    display_profile(analysis['profile'])
    display_price_data(analysis['price_data'])
    display_valuation(analysis['valuation'])
    display_profitability(analysis['profitability'])
    display_financial_health(analysis['financial_health'])
    display_growth(analysis['growth'])
    
    if analysis['dividends']['dividend_yield'] > 0:
        display_dividends(analysis['dividends'])
    
    # Full analysis
    if full_analysis:
        print(f"\n{Colors.CYAN}Computing advanced analysis...{Colors.END}")
        
        # DCF
        try:
            dcf_result = _calculate_dcf(analyzer)
            display_dcf(dcf_result)
        except Exception as e:
            logger.warning(f"DCF failed: {e}")
        
        # Risk metrics
        try:
            risk = _calculate_risk_metrics(ticker)
            display_risk(risk)
        except Exception as e:
            logger.warning(f"Risk analysis failed: {e}")
        
        # Technical analysis
        try:
            technical = _calculate_technical_analysis(ticker)
            display_technical(technical)
        except Exception as e:
            logger.warning(f"Technical analysis failed: {e}")
        
        # Summary
        full_data = {
            "ticker": ticker.upper(),
            "company_name": analysis['profile']['name'],
            "fundamental_analysis": analysis,
            "valuation": dcf_result if 'dcf_result' in locals() else None,
            "risk_metrics": risk if 'risk' in locals() else None,
            "technical_analysis": technical if 'technical' in locals() else None
        }
        
        summary = _generate_summary(full_data)
        display_summary(summary)
        
        # Export if requested
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    json.dump(full_data, f, indent=2, default=str)
                print(f"\n{Colors.GREEN}✓ Analysis exported to {export_path}{Colors.END}")
            except Exception as e:
                print(f"\n{Colors.RED}Failed to export: {e}{Colors.END}")


def display_profile(profile: Dict):
    """Display company profile."""
    print_section("COMPANY PROFILE")
    print_metric("Name", profile['name'])
    print_metric("Ticker", profile['ticker'])
    print_metric("Sector", profile['sector'])
    print_metric("Industry", profile['industry'])
    print_metric("Country", profile['country'])
    print_metric("Employees", f"{profile['employees']:,}" if profile['employees'] else "N/A")
    print_metric("Market Cap", format_currency(profile['market_cap']))


def display_price_data(data: Dict):
    """Display price data."""
    print_section("PRICE DATA")
    current = data['current_price']
    prev = data['previous_close']
    change = ((current - prev) / prev * 100) if prev else 0
    color = Colors.GREEN if change >= 0 else Colors.RED
    
    print_metric("Current Price", f"${current:.2f}", color)
    print_metric("Change", f"{change:+.2f}%", color)
    print_metric("52-Week Range", f"${data['52_week_low']:.2f} - ${data['52_week_high']:.2f}")
    print_metric("Volume", f"{data['volume']:,}")
    print_metric("Avg Volume", f"{data['avg_volume']:,}")
    print_metric("Beta", format_number(data['beta']))


def display_valuation(valuation: Dict):
    """Display valuation metrics."""
    print_section("VALUATION")
    print_metric("P/E Ratio", format_number(valuation['pe_ratio']))
    print_metric("Forward P/E", format_number(valuation['forward_pe']))
    print_metric("PEG Ratio", format_number(valuation['peg_ratio']))
    print_metric("Price/Book", format_number(valuation['price_to_book']))
    print_metric("Price/Sales", format_number(valuation['price_to_sales']))
    print_metric("EV/Revenue", format_number(valuation['ev_to_revenue']))
    print_metric("EV/EBITDA", format_number(valuation['ev_to_ebitda']))


def display_profitability(profit: Dict):
    """Display profitability metrics."""
    print_section("PROFITABILITY")
    print_metric("Gross Margin", format_percentage(profit['gross_margin']))
    print_metric("Operating Margin", format_percentage(profit['operating_margin']))
    print_metric("Net Profit Margin", format_percentage(profit['net_profit_margin']))
    print_metric("EBITDA Margin", format_percentage(profit['ebitda_margin']))
    print_metric("ROE", format_percentage(profit['roe']))
    print_metric("ROA", format_percentage(profit['roa']))
    print_metric("ROIC", format_percentage(profit['roic']))


def display_financial_health(health: Dict):
    """Display financial health metrics."""
    print_section("FINANCIAL HEALTH")
    print_metric("Current Ratio", format_number(health['current_ratio']))
    print_metric("Quick Ratio", format_number(health['quick_ratio']))
    print_metric("Debt/Equity", format_number(health['debt_to_equity']))
    print_metric("Interest Coverage", format_number(health['interest_coverage']))
    print_metric("Total Cash", format_currency(health['total_cash']))
    print_metric("Total Debt", format_currency(health['total_debt']))
    print_metric("Net Debt", format_currency(health['net_debt']))


def display_growth(growth: Dict):
    """Display growth metrics."""
    print_section("GROWTH")
    print_metric("Revenue Growth (YoY)", format_percentage(growth['revenue_growth_yoy']))
    print_metric("Revenue Growth (QoQ)", format_percentage(growth['revenue_growth_qoq']))
    print_metric("Earnings Growth (YoY)", format_percentage(growth['earnings_growth_yoy']))
    print_metric("Revenue CAGR (3Y)", format_percentage(growth['revenue_cagr_3y']))
    print_metric("Earnings CAGR (3Y)", format_percentage(growth['earnings_cagr_3y']))


def display_dividends(dividends: Dict):
    """Display dividend information."""
    print_section("DIVIDENDS")
    print_metric("Dividend Yield", format_percentage(dividends['dividend_yield']))
    print_metric("Dividend Rate", f"${dividends['dividend_rate']:.2f}")
    print_metric("Payout Ratio", format_percentage(dividends['payout_ratio']))
    print_metric("5Y Avg Yield", format_percentage(dividends['five_year_avg_yield']))


def display_dcf(dcf: Dict):
    """Display DCF valuation."""
    print_section("DCF VALUATION")
    
    if 'error' in dcf:
        print(f"{Colors.YELLOW}  DCF calculation unavailable: {dcf['error']}{Colors.END}")
        return
    
    intrinsic = dcf['value_per_share']
    current = dcf['current_price']
    upside = dcf['upside']
    
    color = Colors.GREEN if upside > 0 else Colors.RED
    
    print_metric("Intrinsic Value", f"${intrinsic:.2f}")
    print_metric("Current Price", f"${current:.2f}")
    print_metric("Upside/Downside", f"{upside:+.2f}%", color)


def display_risk(risk: Dict):
    """Display risk metrics."""
    print_section("RISK METRICS")
    
    if 'error' in risk:
        print(f"{Colors.YELLOW}  Risk analysis unavailable: {risk['error']}{Colors.END}")
        return
    
    print_metric("Annual Volatility", format_percentage(risk['volatility_annual']))
    print_metric("Sharpe Ratio", format_number(risk['sharpe_ratio']))
    print_metric("Max Drawdown", format_percentage(risk['max_drawdown']), Colors.RED)
    print_metric("VaR (95%)", format_percentage(risk['var_95']), Colors.YELLOW)
    print_metric("CVaR (95%)", format_percentage(risk['cvar_95']), Colors.RED)


def display_technical(technical: Dict):
    """Display technical analysis."""
    print_section("TECHNICAL ANALYSIS")
    
    if 'error' in technical:
        print(f"{Colors.YELLOW}  Technical analysis unavailable: {technical['error']}{Colors.END}")
        return
    
    trend_color = Colors.GREEN if technical['trend_short_term'] == 'bullish' else Colors.RED
    
    print_metric("Current Price", f"${technical['current_price']:.2f}")
    print_metric("MA(20)", f"${technical['ma_20']:.2f}")
    print_metric("MA(50)", f"${technical['ma_50']:.2f}")
    if technical['ma_200']:
        print_metric("MA(200)", f"${technical['ma_200']:.2f}")
    print_metric("RSI", format_number(technical['rsi']))
    print_metric("Short-term Trend", technical['trend_short_term'], trend_color)
    
    if technical['signals']:
        print(f"\n  {Colors.BOLD}Signals:{Colors.END} {', '.join(technical['signals'])}")


def display_summary(summary: Dict):
    """Display analysis summary."""
    print_section("ANALYSIS SUMMARY")
    
    score = summary['overall_score']
    grade = summary['overall_grade']
    
    # Color based on grade
    if score >= 70:
        grade_color = Colors.GREEN
    elif score >= 50:
        grade_color = Colors.YELLOW
    else:
        grade_color = Colors.RED
    
    print_metric("Overall Score", f"{score:.1f}/100", grade_color)
    print_metric("Overall Grade", grade, grade_color)
    
    print(f"\n  {Colors.BOLD}Component Grades:{Colors.END}")
    print_metric("  • Valuation", summary['valuation_grade']['letter'])
    print_metric("  • Profitability", summary['profitability_grade']['letter'])
    print_metric("  • Financial Health", summary['financial_health_grade']['letter'])
    print_metric("  • Growth", summary['growth_grade']['letter'])
    
    rec = summary['recommendation']
    rec_color = Colors.GREEN if 'Buy' in rec['rating'] else Colors.YELLOW if 'Hold' in rec['rating'] else Colors.RED
    
    print(f"\n  {Colors.BOLD}Investment Recommendation:{Colors.END}")
    print_metric("  Rating", rec['rating'], rec_color)
    print_metric("  Confidence", rec['confidence'])
    
    if rec['key_strengths']:
        print(f"\n  {Colors.BOLD}{Colors.GREEN}Key Strengths:{Colors.END}")
        for strength in rec['key_strengths']:
            print(f"    ✓ {strength}")
    
    if rec['key_risks']:
        print(f"\n  {Colors.BOLD}{Colors.RED}Key Risks:{Colors.END}")
        for risk in rec['key_risks']:
            print(f"    ⚠ {risk}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Company Analyzer - Search and analyze any company",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_company.py                    # Interactive search
  python analyze_company.py TSLA                # Analyze Tesla
  python analyze_company.py --search "Apple"   # Search for Apple
  python analyze_company.py AAPL --full --export report.json
        """
    )
    
    parser.add_argument('ticker', nargs='?', help='Stock ticker symbol')
    parser.add_argument('--search', '-s', help='Search for company by name')
    parser.add_argument('--full', '-f', action='store_true', help='Full analysis (DCF, risk, technical)')
    parser.add_argument('--export', '-e', help='Export results to JSON file')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick analysis (fundamentals only)')
    
    args = parser.parse_args()
    
    # Determine ticker
    ticker = None
    
    if args.search:
        # Search mode
        searcher = CompanySearch()
        results = searcher.search(args.search, limit=5)
        if results:
            print(f"{Colors.GREEN}Found: {results[0]['ticker']} - {results[0]['name']}{Colors.END}")
            ticker = results[0]['ticker']
        else:
            print(f"{Colors.RED}No results found for '{args.search}'{Colors.END}")
            return
    
    elif args.ticker:
        # Direct ticker
        ticker = args.ticker.upper()
    
    else:
        # Interactive mode
        ticker = search_interactive()
    
    # Analyze if ticker selected
    if ticker:
        full = args.full or not args.quick
        analyze_company(ticker, full_analysis=full, export_path=args.export)
    else:
        print(f"\n{Colors.YELLOW}No company selected{Colors.END}")


if __name__ == "__main__":
    main()
