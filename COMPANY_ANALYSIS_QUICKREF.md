# Company Analysis - Quick Reference

## 🚀 Quick Commands

### CLI Usage
```bash
# Interactive mode
python analyze_company.py

# Analyze specific ticker
python analyze_company.py TSLA
python analyze_company.py AAPL --full
python analyze_company.py MSFT --quick

# Search by name
python analyze_company.py --search "Tesla"
python analyze_company.py --search "Apple"

# Export results
python analyze_company.py TSLA --full --export tesla_report.json
```

### API Usage (with server running)
```bash
# Search companies
curl "http://localhost:8000/api/v1/company/search?query=Tesla"

# Analyze company
curl "http://localhost:8000/api/v1/company/analyze/TSLA"

# Get sector companies
curl "http://localhost:8000/api/v1/company/sector/Technology"

# Get top companies
curl "http://localhost:8000/api/v1/company/top-companies?n=50"
```

## 📊 Python Examples

### Basic Analysis
```python
from models.fundamental.company_analyzer import CompanyAnalyzer

analyzer = CompanyAnalyzer("TSLA")
analysis = analyzer.comprehensive_analysis()

print(analysis['profile'])        # Company info
print(analysis['valuation'])      # P/E, P/B, etc.
print(analysis['profitability'])  # Margins, ROE
print(analysis['financial_health']) # Debt, liquidity
```

### Search Companies
```python
from core.company_search import CompanySearch, search_companies

# Quick search
results = search_companies("Tesla", limit=5)

# Advanced search
searcher = CompanySearch()
tech_companies = searcher.filter_by_sector("Technology")
large_caps = searcher.filter_by_market_cap(min_cap=100e9)
top_50 = searcher.get_top_companies(n=50)
```

### Complete Analysis with Grading
```python
from models.fundamental.company_analyzer import CompanyAnalyzer
from api.company_analysis_api import (
    _calculate_dcf,
    _calculate_risk_metrics,
    _calculate_technical_analysis,
    _generate_summary
)

# Analyze
analyzer = CompanyAnalyzer("AAPL")
analysis = analyzer.comprehensive_analysis()
dcf = _calculate_dcf(analyzer)
risk = _calculate_risk_metrics("AAPL")
technical = _calculate_technical_analysis("AAPL")

# Get summary with grades
full_data = {
    "ticker": "AAPL",
    "company_name": analysis['profile']['name'],
    "fundamental_analysis": analysis,
    "valuation": dcf,
    "risk_metrics": risk,
    "technical_analysis": technical
}

summary = _generate_summary(full_data)
print(f"Overall Grade: {summary['overall_grade']}")
print(f"Recommendation: {summary['recommendation']['rating']}")
```

### Batch Analysis
```python
portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

for ticker in portfolio:
    analyzer = CompanyAnalyzer(ticker)
    analysis = analyzer.comprehensive_analysis()
    
    print(f"\n{ticker}:")
    print(f"  P/E: {analysis['valuation']['pe_ratio']:.2f}")
    print(f"  ROE: {analysis['profitability']['roe']:.2%}")
    print(f"  Debt/Equity: {analysis['financial_health']['debt_to_equity']:.2f}")
```

### Custom Screening
```python
from core.company_search import CompanySearch
from models.fundamental.company_analyzer import CompanyAnalyzer

searcher = CompanySearch()
companies = searcher.get_top_companies(n=100)

# Find undervalued growth stocks
for company in companies:
    ticker = company['ticker']
    try:
        analyzer = CompanyAnalyzer(ticker)
        analysis = analyzer.comprehensive_analysis()
        
        pe = analysis['valuation'].get('pe_ratio')
        growth = analysis['growth'].get('revenue_growth_yoy')
        roe = analysis['profitability'].get('roe')
        
        # Screen criteria
        if pe and growth and roe:
            if pe < 20 and growth > 0.15 and roe > 0.15:
                print(f"✓ {ticker}: P/E={pe:.1f}, Growth={growth:.1%}, ROE={roe:.1%}")
    except:
        pass
```

## 📈 Analysis Components

### Profile
- name, ticker, sector, industry, country
- employees, market_cap, enterprise_value

### Price Data
- current_price, previous_close, day_high/low
- 52_week_high/low, volume, avg_volume, beta

### Valuation
- pe_ratio, forward_pe, peg_ratio
- price_to_book, price_to_sales
- ev_to_revenue, ev_to_ebitda

### Profitability
- gross_margin, operating_margin, net_profit_margin
- ebitda_margin, roe, roa, roic

### Financial Health
- current_ratio, quick_ratio
- debt_to_equity, debt_to_assets
- interest_coverage, cash_to_debt

### Growth
- revenue_growth_yoy, revenue_growth_qoq
- earnings_growth_yoy, earnings_growth_qoq
- revenue_cagr_3y, earnings_cagr_3y

### DCF Valuation
- intrinsic_value, value_per_share
- current_price, upside

### Risk Metrics
- var_95, var_99, cvar_95, cvar_99
- volatility_annual, max_drawdown, sharpe_ratio

### Technical Analysis
- ma_20, ma_50, ma_200
- rsi, trend_short_term, trend_long_term
- signals (overbought, oversold, crossovers)

### Summary & Grading
- valuation_grade, profitability_grade
- financial_health_grade, growth_grade
- overall_score, overall_grade
- recommendation (rating, key_strengths, key_risks)

## 🎯 Common Use Cases

### 1. Quick Company Check
```bash
python analyze_company.py AAPL --quick
```

### 2. Deep Dive Analysis
```bash
python analyze_company.py TSLA --full --export tesla_full.json
```

### 3. Portfolio Monitoring
```python
portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN"]
for ticker in portfolio:
    analyzer = CompanyAnalyzer(ticker)
    analysis = analyzer.comprehensive_analysis()
    print(f"{ticker}: {analysis['price_data']['current_price']}")
```

### 4. Sector Analysis
```python
searcher = CompanySearch()
tech = searcher.filter_by_sector("Technology")
print(f"Found {len(tech)} tech companies")
```

### 5. Value Screening
```python
searcher = CompanySearch()
for company in searcher.get_top_companies(50):
    analyzer = CompanyAnalyzer(company['ticker'])
    analysis = analyzer.comprehensive_analysis()
    if analysis['valuation']['pe_ratio'] < 15:
        print(f"{company['ticker']}: P/E={analysis['valuation']['pe_ratio']}")
```

## 🔧 Tips & Tricks

### Fuzzy Search
```python
# All of these work
search_companies("Apple")
search_companies("AAPL")
search_companies("apple computer")
search_companies("apl")  # Close match
```

### Error Handling
```python
from core.company_search import CompanySearch

searcher = CompanySearch()
is_valid, message = searcher.validate_ticker("TSLA")
if is_valid:
    # Proceed with analysis
    pass
```

### Caching
- Company database cached for 7 days
- Stock data cached for 5 minutes (from DataFetcher)
- Cache location: `data/cache/`

### Export Formats
```bash
# JSON (detailed)
python analyze_company.py TSLA --export report.json

# Then process with Python
import json
with open('report.json') as f:
    data = json.load(f)
```

## 📚 Documentation

- **Full Guide**: `COMPANY_ANALYSIS_GUIDE.md`
- **Implementation**: `COMPANY_ANALYSIS_IMPLEMENTATION.md`
- **API Docs**: `http://localhost:8000/docs` (when API running)

## ⚡ Performance

- Search: < 1 second (cached)
- Basic analysis: ~5 seconds
- Full analysis: ~10-15 seconds
- API response: ~10 seconds

## 🐛 Troubleshooting

**Module not found**
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt  # For API
```

**No data available**
- Check ticker is valid
- Try different period: `period="2y"`
- Verify internet connection

**API not working**
```bash
# Start server
python -m uvicorn api.main:app --reload

# Check health
curl http://localhost:8000/health
```

## 📞 Quick Help

```bash
# CLI help
python analyze_company.py --help

# API documentation
http://localhost:8000/docs  # (with API running)
```
