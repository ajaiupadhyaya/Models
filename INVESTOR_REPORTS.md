# Investor Report Generation with OpenAI

Professional, narrative-driven investor reports powered by GPT-4 and GPT-3.5-turbo.

## Overview

The investor report generator creates institutional-quality communications suitable for:
- **Quarterly investor updates** - Comprehensive performance summaries
- **Fund fact sheets** - Key metrics and strategy descriptions
- **Investor communications** - Narrative-driven performance analysis
- **Due diligence documentation** - Risk analysis and forward-looking statements

## Architecture

### Core Components

**1. InvestorReportGenerator** (`core/investor_reports.py`)
```python
from core.investor_reports import InvestorReportGenerator

# Initialize with OpenAI API key
generator = InvestorReportGenerator()

# Generate full report
report = generator.generate_full_report(
    title="Q2 2024 Investor Update",
    models=[...],
    backtest_results=[...],
    # ... additional parameters
)

# Export to file
generator.export_report_to_markdown(report, "report.md")
generator.export_report_to_html(report, "report.html")
```

**Key Methods:**
- `generate_executive_summary()` - 300-400 word strategic overview
- `generate_strategy_overview()` - 400-500 word strategy explanation
- `generate_research_findings()` - 500-600 word proprietary insights
- `generate_performance_analysis()` - 600-800 word detailed analysis
- `generate_risk_analysis()` - 400-500 word risk assessment
- `generate_forward_looking_statements()` - 300-400 word outlook
- `generate_full_report()` - Orchestrates all sections
- `export_report_to_markdown()` - Saves readable markdown
- `export_report_to_html()` - Saves professional HTML

**2. Investor Reports API** (`api/investor_reports_api.py`)

REST API endpoints for report generation:

```bash
# Generate comprehensive investor report
POST /api/v1/reports/generate
{
  "title": "Q2 2024 Update",
  "models": [...],
  "backtest_results": [...],
  "export_format": "markdown" | "html"
}

# Generate quick investor update
POST /api/v1/reports/quick-update
{
  "title": "Flash Update",
  "key_metrics": {...},
  "highlights": [...],
  "risks": [...]
}

# Get report examples
GET /api/v1/reports/examples

# Check API health
GET /api/v1/reports/health
```

## Setup

### 1. Install OpenAI SDK

```bash
pip install openai>=1.0.0
```

### 2. Configure API Key

```bash
export OPENAI_API_KEY="sk-..."
```

Or set in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### 3. Verify Configuration

```python
from core.investor_reports import InvestorReportGenerator

gen = InvestorReportGenerator()
if gen.client:
    print("✓ OpenAI API configured")
else:
    print("✗ API key not found")
```

## Usage Examples

### Example 1: Generate Full Report

```python
from core.investor_reports import (
    InvestorReportGenerator,
    ModelPerformance,
    BacktestResults
)

# Initialize
generator = InvestorReportGenerator()

# Create model performance objects
models = [
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
    )
]

# Create backtest results
backtest_results = [
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
            "cvar_95": -2.8
        },
        trading_metrics={
            "avg_trade_duration_days": 8.2,
            "consecutive_wins": 12
        }
    )
]

# Generate report
report = generator.generate_full_report(
    title="Q2 2024 Investor Update",
    models=models,
    backtest_results=backtest_results,
    strategy_descriptions={
        "Ensemble ML Strategy": "Multi-model ensemble..."
    },
    research_findings=[
        {
            "title": "Key Discovery",
            "description": "Details of the finding"
        }
    ],
    risk_metrics={"portfolio_var_95": -2.4},
    market_outlook="Balanced with selective opportunities",
    period="Q2 2024",
    fund_name="Trading ML Fund"
)

# Export to files
generator.export_report_to_markdown(report, "report.md")
generator.export_report_to_html(report, "report.html")

print(f"Report generated: {report.title}")
print(f"Files saved to data/reports/")
```

### Example 2: API Usage

```bash
# Start the API server
python api/main.py

# In another terminal
curl -X POST http://localhost:8000/api/v1/reports/generate \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Q2 2024 Update",
    "models": [...],
    "backtest_results": [...],
    "export_format": "html"
  }'
```

### Example 3: Quick Investor Update

```python
update = generator._call_gpt(
    prompt="""
    Write a professional investor update (200-300 words):
    - Highlight key performance metrics
    - Discuss strategic positioning
    - Address major risks
    - Outline next steps
    """,
    max_tokens=1000
)

print(update)
```

## Report Structure

### 8-Section Professional Framework

1. **Executive Summary** (300-400 words)
   - Key metrics and performance highlights
   - Strategic positioning overview
   - Competitive advantages

2. **Strategy Overview** (400-500 words)
   - Core philosophy and approach
   - Mechanical explanation of strategies
   - Model architecture and inputs

3. **Research Findings** (500-600 words)
   - Proprietary discoveries
   - Quantitative evidence
   - Differentiated insights

4. **Performance Analysis** (600-800 words)
   - Detailed performance metrics
   - Comparison vs benchmarks
   - Attribution analysis
   - Sharpe/Sortino/Calmar ratios

5. **Risk Analysis** (400-500 words)
   - Value at Risk (VaR)
   - Conditional Value at Risk (CVaR)
   - Stress test results
   - Correlation analysis

6. **Forward-Looking Statements** (300-400 words)
   - Market outlook
   - Strategic adjustments
   - Competitive positioning
   - Risk disclosures

7. **Appendix** (Data Tables)
   - Model performance summary
   - Backtest results table
   - Monthly returns
   - Key statistics

8. **Disclaimer** (Legal)
   - Past performance disclaimer
   - Risk disclosures
   - Important notices
   - Regulatory compliance

## Export Formats

### Markdown
- Human-readable format
- Suitable for documentation
- Easy version control
- Lightweight file size

### HTML
- Professional styling with CSS
- Print-ready layout
- Interactive elements ready
- Suitable for web distribution
- Color-coded sections
- Professional typography

## Data Models

### ModelPerformance
```python
@dataclass
class ModelPerformance:
    name: str
    type: str  # "Simple ML", "Ensemble", "LSTM", etc.
    total_return: float  # %
    annual_return: float  # %
    sharpe_ratio: float
    max_drawdown: float  # %
    win_rate: float  # %
    trades_count: int
    best_trade: float  # %
    worst_trade: float  # %
    avg_trade_return: float  # %
    profit_factor: float
    recovery_factor: float
```

### BacktestResults
```python
@dataclass
class BacktestResults:
    symbol: str
    period: str
    model_performance: ModelPerformance
    market_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    monthly_returns: Optional[Dict[str, float]]
```

### InvestorReport
```python
@dataclass
class InvestorReport:
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
```

## Notebook Example

See `notebooks/07_investor_reports.ipynb` for complete working example:
- Initialization and configuration
- Sample data creation
- Full report generation
- Export to multiple formats
- API usage examples

Run with:
```bash
jupyter notebook notebooks/07_investor_reports.ipynb
```

## API Endpoints

### POST /api/v1/reports/generate
Generate comprehensive investor report.

**Request:**
```json
{
  "title": "Q2 2024 Investor Update",
  "models": [...],
  "backtest_results": [...],
  "strategy_descriptions": {...},
  "research_findings": [...],
  "risk_metrics": {...},
  "var_metrics": {...},
  "market_outlook": "...",
  "strategy_adjustments": [...],
  "period": "Q2 2024",
  "fund_name": "Trading ML Fund",
  "export_format": "markdown"
}
```

**Response:**
```json
{
  "title": "Q2 2024 Investor Update",
  "date": "2024-06-30T12:00:00",
  "executive_summary": "...",
  "strategy_overview": "...",
  "research_findings": "...",
  "performance_analysis": "...",
  "risk_analysis": "...",
  "forward_looking_statements": "...",
  "appendix": "...",
  "disclaimer": "...",
  "file_path": "data/reports/investor_report_20240630_120000.md"
}
```

### POST /api/v1/reports/quick-update
Generate rapid investor update (200-300 words).

### POST /api/v1/reports/performance-summary
Generate performance summary for single strategy.

### POST /api/v1/reports/analyze-backtest
Generate narrative analysis of backtest results.

### GET /api/v1/reports/examples
List available report templates and examples.

### GET /api/v1/reports/health
Check API health and OpenAI configuration.

## Configuration

### Environment Variables

```bash
# OpenAI API Key (required)
export OPENAI_API_KEY="sk-..."

# Optional: Customize models and parameters
# Edit core/investor_reports.py to change:
# - Default model (GPT-4, GPT-3.5-turbo)
# - Temperature (0.7 for balanced output)
# - Max tokens per section
```

### Customization

Modify `InvestorReportGenerator` class for:
- Different GPT models
- Custom system prompts
- Report section lengths
- HTML styling
- Export formats

## Performance & Costs

### API Costs (Approximate)
- GPT-4-turbo: ~$0.15-0.20 per full report
- GPT-3.5-turbo: ~$0.02-0.03 per full report

### Generation Time
- Full report: 1-2 minutes (6 sections)
- Quick update: 15-30 seconds
- Single section: 10-20 seconds

### Optimization
- Use GPT-3.5-turbo for cost-sensitive deployments
- Batch generate reports during off-peak hours
- Cache frequently generated sections
- Implement report templates

## Troubleshooting

### OpenAI API Key Not Found
```
Solution: Set OPENAI_API_KEY environment variable
export OPENAI_API_KEY="sk-..."
```

### Rate Limit Errors
```
Solution: Wait before retrying, use exponential backoff
# Generator implements automatic retry logic
```

### Invalid Data in Report
```
Solution: Verify ModelPerformance and BacktestResults objects
# Check that all required fields are populated with valid numbers
```

## Best Practices

1. **Data Quality**
   - Ensure all metrics are calculated correctly
   - Use realistic sample data for testing
   - Validate data before report generation

2. **Report Frequency**
   - Generate weekly for trading updates
   - Generate monthly for investor communications
   - Generate quarterly for formal reports

3. **Customization**
   - Adjust report sections as needed
   - Include fund-specific terminology
   - Add market context to findings

4. **Distribution**
   - Use HTML for web/email distribution
   - Use Markdown for documentation
   - Include proper disclaimers
   - Version reports with timestamps

## Advanced Features

### Scheduled Report Generation
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(generate_weekly_report, 'cron', day_of_week='fri', hour=17)
scheduler.start()
```

### Report Templates
```python
# Create custom templates for different investor types
investor_types = ["institutional", "retail", "hedge_fund"]

for investor_type in investor_types:
    report = generator.generate_full_report(..., template=investor_type)
```

### Report Caching
```python
# Cache frequently generated reports
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_cached_report(symbol, period):
    return generator.generate_full_report(...)
```

## Integration Examples

### With Backtesting Engine
```python
from models.trading.backtesting import BacktestEngine

engine = BacktestEngine()
results = engine.run_backtest(strategy, data)

# Convert to BacktestResults for report
backtest_report = BacktestResults(
    symbol=results.symbol,
    period=results.period,
    # ... map results to report format
)

# Generate investor report
report = generator.generate_full_report(backtest_results=[backtest_report])
```

### With FastAPI
```python
from api.investor_reports_api import router

app.include_router(router, prefix="/api/v1/reports", tags=["Reports"])

# Now available at /api/v1/reports/generate
```

## Next Steps

1. Set OpenAI API key
2. Run `notebooks/07_investor_reports.ipynb` for examples
3. Integrate with your backtesting data
4. Deploy API server with `python api/main.py`
5. Schedule automated report generation
6. Customize templates for your fund
7. Distribute reports to investors

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [GPT-4 Models](https://platform.openai.com/docs/models/gpt-4)
- [Investment Report Best Practices](https://www.cfainstitute.org)
- [Financial Disclosure Standards](https://www.sec.gov)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `notebooks/07_investor_reports.ipynb`
3. Check API logs: `tail -f logs/api.log`
4. Verify OpenAI API status and account
