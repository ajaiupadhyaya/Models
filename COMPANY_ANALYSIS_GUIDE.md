# Company Analysis System

## Overview

The Company Analysis System provides comprehensive automated analysis for any publicly traded company. You can search for companies, validate tickers, and receive detailed analysis including:

- **Fundamental Analysis**: Financial statements, ratios, profitability, efficiency
- **Valuation Metrics**: DCF valuation, multiples, intrinsic value
- **Risk Analysis**: VaR, CVaR, volatility, drawdown, Sharpe ratio
- **Technical Analysis**: Moving averages, RSI, trend signals
- **Investment Ratings**: Automated grading and recommendations

## Quick Start

### 1. Command Line Interface (CLI)

The easiest way to analyze a company:

```bash
# Interactive search and analysis
python analyze_company.py

# Direct ticker analysis
python analyze_company.py TSLA

# Search by company name
python analyze_company.py --search "Tesla"

# Full analysis with export
python analyze_company.py AAPL --full --export tesla_report.json

# Quick fundamentals only
python analyze_company.py MSFT --quick
```

#### CLI Features

- **Interactive Search**: Fuzzy matching for company names and tickers
- **Comprehensive Analysis**: All metrics in one place
- **Color-coded Output**: Easy-to-read terminal display
- **Export Options**: Save results to JSON for further processing

### 2. API Endpoints

The company analysis system is integrated into the FastAPI server:

```bash
# Start the API server
python -m uvicorn api.main:app --reload
```

#### Available Endpoints

**Search for Companies**
```http
GET /api/v1/company/search?query=Tesla&limit=10
```

**Validate Ticker**
```http
GET /api/v1/company/validate/TSLA
```

**Complete Company Analysis**
```http
GET /api/v1/company/analyze/TSLA?include_dcf=true&include_risk=true&include_technicals=true
```

**Get Companies by Sector**
```http
GET /api/v1/company/sector/Technology?limit=50
```

**Get Top Companies**
```http
GET /api/v1/company/top-companies?n=50
```

**Get Available Sectors**
```http
GET /api/v1/company/sectors
```

### 3. Python API

Use the system programmatically in your Python scripts:

```python
from core.company_search import CompanySearch, search_companies
from models.fundamental.company_analyzer import CompanyAnalyzer
from api.company_analysis_api import (
    _calculate_dcf,
    _calculate_risk_metrics,
    _calculate_technical_analysis,
    _generate_summary
)

# Search for companies
results = search_companies("Tesla", limit=5)
print(results)

# Get company by ticker
searcher = CompanySearch()
company = searcher.get_by_ticker("TSLA")

# Comprehensive analysis
analyzer = CompanyAnalyzer("TSLA")
analysis = analyzer.comprehensive_analysis()

# Profile
print(analysis['profile'])

# Valuation metrics
print(analysis['valuation'])

# Financial health
print(analysis['financial_health'])

# DCF Valuation
dcf_result = _calculate_dcf(analyzer)
print(f"Intrinsic Value: ${dcf_result['value_per_share']:.2f}")
print(f"Current Price: ${dcf_result['current_price']:.2f}")
print(f"Upside: {dcf_result['upside']:.2f}%")

# Risk metrics
risk = _calculate_risk_metrics("TSLA", period="1y")
print(f"Annual Volatility: {risk['volatility_annual']:.2%}")
print(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {risk['max_drawdown']:.2%}")

# Technical analysis
technical = _calculate_technical_analysis("TSLA", period="1y")
print(f"Trend: {technical['trend_short_term']}")
print(f"RSI: {technical['rsi']:.2f}")

# Summary with grades
summary = _generate_summary({
    "ticker": "TSLA",
    "company_name": analysis['profile']['name'],
    "fundamental_analysis": analysis,
    "valuation": dcf_result,
    "risk_metrics": risk,
    "technical_analysis": technical
})
print(f"Overall Grade: {summary['overall_grade']}")
print(f"Recommendation: {summary['recommendation']['rating']}")
```

## Analysis Components

### 1. Company Profile
- Company name, ticker, sector, industry
- Country, employees, website
- Market cap, enterprise value

### 2. Price Data
- Current price, previous close, daily change
- 52-week range, volume, beta
- Day high/low

### 3. Valuation Metrics
- P/E ratio (trailing and forward)
- PEG ratio, Price/Book, Price/Sales
- EV/Revenue, EV/EBITDA
- Market cap, shares outstanding, short interest

### 4. Profitability Metrics
- Gross, operating, net profit margins
- EBITDA margin
- ROE, ROA, ROIC

### 5. Financial Health
- Current ratio, quick ratio
- Debt/Equity, Debt/Assets
- Interest coverage
- Cash position, net debt

### 6. Growth Metrics
- Revenue growth (YoY, QoQ)
- Earnings growth (YoY, QoQ)
- CAGR (3-year)

### 7. Dividend Information
- Dividend yield, dividend rate
- Payout ratio
- Ex-dividend date

### 8. Efficiency Metrics
- Asset turnover
- Inventory turnover
- Receivables turnover
- Cash conversion cycle

### 9. DCF Valuation
- Intrinsic value calculation
- Enterprise value
- Value per share
- Upside/downside vs current price

### 10. Risk Analysis
- Value at Risk (VaR) at 95% and 99%
- Conditional VaR (CVaR)
- Annual volatility
- Maximum drawdown
- Sharpe ratio

### 11. Technical Analysis
- Moving averages (20, 50, 200-day)
- RSI (Relative Strength Index)
- Trend identification
- Trading signals (overbought, oversold, crossovers)

### 12. Investment Summary
- Component grades (Valuation, Profitability, Health, Growth)
- Overall score and grade
- Investment recommendation (Buy/Hold/Sell)
- Key strengths and risks
- Confidence level

## Usage Examples

### Example 1: Analyze Tesla

```bash
python analyze_company.py TSLA --full
```

**Output includes:**
- Company profile (Tesla Inc., automotive, employees, market cap)
- Current stock price and 52-week range
- Valuation: P/E 65, PEG 2.1, Price/Book 12
- Profitability: Net margin 15%, ROE 28%
- Financial health: Current ratio 1.8, Debt/Equity 0.3
- DCF intrinsic value vs current price
- Risk: Volatility 45%, Sharpe 1.2, Max DD -35%
- Technical: RSI 62, bullish trend
- **Overall Grade: B+ (Strong Buy)**

### Example 2: Compare Companies

```python
companies = ["AAPL", "MSFT", "GOOGL", "AMZN"]

for ticker in companies:
    analyzer = CompanyAnalyzer(ticker)
    analysis = analyzer.comprehensive_analysis()
    
    print(f"\n{ticker}:")
    print(f"  P/E: {analysis['valuation']['pe_ratio']:.2f}")
    print(f"  ROE: {analysis['profitability']['roe']:.2%}")
    print(f"  Revenue Growth: {analysis['growth']['revenue_growth_yoy']:.2%}")
```

### Example 3: Screen by Criteria

```python
from core.company_search import CompanySearch

searcher = CompanySearch()

# Get tech companies
tech_companies = searcher.filter_by_sector("Technology")

# Filter by market cap > $100B
large_caps = searcher.filter_by_market_cap(min_cap=100e9)

# Analyze top 10
for company in large_caps[:10]:
    ticker = company['ticker']
    analyzer = CompanyAnalyzer(ticker)
    analysis = analyzer.comprehensive_analysis()
    
    # Check criteria
    roe = analysis['profitability']['roe']
    pe = analysis['valuation']['pe_ratio']
    
    if roe and roe > 0.15 and pe and pe < 30:
        print(f"✓ {ticker}: ROE={roe:.2%}, P/E={pe:.2f}")
```

### Example 4: API Usage with curl

```bash
# Search for a company
curl "http://localhost:8000/api/v1/company/search?query=Apple&limit=5"

# Get full analysis
curl "http://localhost:8000/api/v1/company/analyze/AAPL?include_dcf=true&include_risk=true&include_technicals=true"

# Get tech sector companies
curl "http://localhost:8000/api/v1/company/sector/Technology?limit=20"

# Get top 50 companies by market cap
curl "http://localhost:8000/api/v1/company/top-companies?n=50"
```

### Example 5: Batch Analysis

```python
import json
from models.fundamental.company_analyzer import CompanyAnalyzer
from api.company_analysis_api import _generate_summary, _calculate_dcf

# List of tickers to analyze
portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

results = []

for ticker in portfolio:
    try:
        analyzer = CompanyAnalyzer(ticker)
        analysis = analyzer.comprehensive_analysis()
        dcf = _calculate_dcf(analyzer)
        
        full_analysis = {
            "ticker": ticker,
            "company_name": analysis['profile']['name'],
            "fundamental_analysis": analysis,
            "valuation": dcf
        }
        
        summary = _generate_summary(full_analysis)
        
        results.append({
            "ticker": ticker,
            "grade": summary['overall_grade'],
            "score": summary['overall_score'],
            "recommendation": summary['recommendation']['rating'],
            "current_price": analysis['price_data']['current_price'],
            "intrinsic_value": dcf.get('value_per_share', 0),
            "upside": dcf.get('upside', 0)
        })
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Save results
with open('portfolio_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

# Display summary
for r in results:
    print(f"{r['ticker']:6s} | Grade: {r['grade']:3s} | {r['recommendation']:12s} | Upside: {r['upside']:+6.1f}%")
```

## Company Search Features

### Fuzzy Matching

The search system uses fuzzy string matching to find companies even with partial or misspelled names:

```python
from core.company_search import search_companies

# All of these work:
search_companies("Apple")
search_companies("AAPL")
search_companies("apple computer")
search_companies("apl")  # Close match to AAPL

# Results include match scores
results = search_companies("Microsft")  # Typo
# Still finds "Microsoft Corporation" with high confidence
```

### Filtering

```python
from core.company_search import CompanySearch

searcher = CompanySearch()

# By sector
tech = searcher.filter_by_sector("Technology")
healthcare = searcher.filter_by_sector("Healthcare")

# By market cap
mega_caps = searcher.filter_by_market_cap(min_cap=500e9)
mid_caps = searcher.filter_by_market_cap(min_cap=2e9, max_cap=10e9)

# Top companies
top_50 = searcher.get_top_companies(n=50)
```

### Validation

```python
from core.company_search import CompanySearch

searcher = CompanySearch()

# Check if ticker is valid
is_valid, message = searcher.validate_ticker("TSLA")
print(f"Valid: {is_valid}, Message: {message}")

# Invalid ticker
is_valid, message = searcher.validate_ticker("INVALID")
# Returns: (False, "Ticker INVALID not found")
```

## Grading System

The system provides letter grades (A+ to F) for:

1. **Valuation**: Based on P/E, P/B ratios
2. **Profitability**: Based on margins, ROE, ROA
3. **Financial Health**: Based on liquidity, leverage
4. **Growth**: Based on revenue and earnings growth

**Overall Grade**: Average of all component grades

**Recommendation Scale**:
- **Strong Buy** (Score ≥ 75): Excellent opportunity
- **Buy** (Score 65-74): Good investment
- **Hold** (Score 55-64): Fair, maintain position
- **Underperform** (Score 45-54): Consider reducing
- **Sell** (Score < 45): Poor fundamentals

## Data Sources

- **Price Data**: Yahoo Finance (yfinance)
- **Fundamentals**: Company financial statements via yfinance
- **Economic Data**: FRED API (optional)
- **Additional**: Alpha Vantage (optional)

## Caching

The company database is automatically cached for 7 days to improve performance:

```python
# Cache location: data/cache/company_database.json
# Automatically refreshed weekly
# Contains ~100+ companies with enriched data
```

## Error Handling

The system gracefully handles errors:

- Invalid tickers return informative messages
- Missing data shows "N/A" instead of crashing
- API errors are logged but don't stop analysis
- Partial results returned if some components fail

## Performance

- **CLI analysis**: 5-15 seconds per company
- **API response**: < 10 seconds for full analysis
- **Search**: < 1 second (cached)
- **Batch analysis**: ~10 seconds per company

## Best Practices

1. **Use caching**: Let the system cache company data
2. **Rate limits**: Be mindful of API rate limits (especially yfinance)
3. **Error handling**: Always wrap API calls in try-except
4. **Validation**: Validate tickers before analyzing
5. **Export**: Save results to JSON for historical tracking

## Troubleshooting

**"No data available"**
- Ticker may be delisted or invalid
- Try with a different period (e.g., "2y" instead of "1y")

**"DCF calculation failed"**
- Company may have negative cash flows
- Financial data may be incomplete

**"Connection error"**
- Check internet connection
- Yahoo Finance may be temporarily unavailable

**"Module not found"**
- Install dependencies: `pip install -r requirements.txt`
- For API: `pip install -r requirements-api.txt`

## Next Steps

1. **Explore the API**: `http://localhost:8000/docs` (when API is running)
2. **Try the CLI**: `python analyze_company.py`
3. **Build custom screens**: Use Python API to create investment screens
4. **Integrate with portfolio**: Connect to your portfolio management system
5. **Automate reports**: Schedule daily/weekly analysis runs

## Support

For issues or questions:
1. Check the API documentation: `/docs`
2. Review example code in this guide
3. Check logs for detailed error messages
4. Validate your API keys (FRED_API_KEY, ALPHA_VANTAGE_API_KEY in `.env`)
