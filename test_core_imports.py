#!/usr/bin/env python3
"""Test core module imports"""

import sys

print('Testing core imports...')

try:
    from core.data_fetcher import DataFetcher
    print('✓ DataFetcher')
except Exception as e:
    print(f'✗ DataFetcher: {e}')

try:
    from core.backtesting import BacktestEngine
    print('✓ BacktestEngine')
except Exception as e:
    print(f'✗ BacktestEngine: {e}')

try:
    from core.investor_reports import InvestorReportGenerator
    print('✓ InvestorReportGenerator')
except Exception as e:
    print(f'✗ InvestorReportGenerator: {e}')

try:
    from core.paper_trading import PaperTradingEngine
    print('✓ PaperTradingEngine')
except Exception as e:
    print(f'✗ PaperTradingEngine: {e}')

try:
    from models.portfolio.optimization import MeanVarianceOptimizer
    print('✓ MeanVarianceOptimizer')
except Exception as e:
    print(f'✗ MeanVarianceOptimizer: {e}')

try:
    from models.valuation.dcf_model import DCFModel
    print('✓ DCFModel')
except Exception as e:
    print(f'✗ DCFModel: {e}')

try:
    from models.risk.var_cvar import VaRModel, CVaRModel, StressTest
    print('✓ VaRModel, CVaRModel, StressTest')
except Exception as e:
    print(f'✗ Risk Models: {e}')

try:
    from models.options.black_scholes import BlackScholes
    print('✓ BlackScholes')
except Exception as e:
    print(f'✗ BlackScholes: {e}')

print('\n✅ Core imports test complete')
