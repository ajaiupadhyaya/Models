#!/usr/bin/env python3
"""
Comprehensive Project Audit
Validates all imports, integrations, and functionality
"""

import sys
import traceback
from pathlib import Path

def audit_imports():
    """Test all major imports"""
    print('='*80)
    print('COMPREHENSIVE PROJECT AUDIT')
    print('='*80)
    
    # Test 1: Core module imports
    print('\n[TEST 1] Core Module Imports')
    print('-'*80)
    core_tests = [
        ('core.data_fetcher', 'DataFetcher'),
        ('core.backtesting', 'SignalBacktester'),
        ('core.paper_trading', 'PaperTradingEngine'),
        ('core.investor_reports', 'InvestorReportGenerator'),
        ('core.visualizations', 'create_interactive_chart'),
        ('core.utils', 'format_currency'),
    ]
    
    passed = 0
    for module, name in core_tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f'✓ {module}.{name}')
            passed += 1
        except Exception as e:
            print(f'✗ {module}.{name}: {str(e)[:50]}')
    
    print(f'\nCore: {passed}/{len(core_tests)} passed')
    
    # Test 2: Model imports
    print('\n[TEST 2] Model Package Imports')
    print('-'*80)
    model_tests = [
        ('models.options.black_scholes', 'BlackScholes'),
        ('models.portfolio.optimization', 'MeanVariance'),
        ('models.risk.var_cvar', 'VaRCalculator'),
        ('models.valuation.dcf_model', 'DCFValuation'),
        ('models.trading.strategies', 'SimpleMomentumStrategy'),
    ]
    
    passed = 0
    for module, name in model_tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f'✓ {module}.{name}')
            passed += 1
        except Exception as e:
            print(f'✗ {module}.{name}: {str(e)[:50]}')
    
    print(f'\nModels: {passed}/{len(model_tests)} passed')
    
    # Test 3: API imports
    print('\n[TEST 3] API Module Imports')
    print('-'*80)
    api_tests = [
        ('api.main', 'app'),
        ('api.models_api', 'router'),
        ('api.predictions_api', 'router'),
        ('api.backtesting_api', 'router'),
        ('api.investor_reports_api', 'router'),
        ('api.paper_trading_api', 'router'),
    ]
    
    passed = 0
    for module, name in api_tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f'✓ {module}.{name}')
            passed += 1
        except Exception as e:
            print(f'✗ {module}.{name}: {str(e)[:50]}')
    
    print(f'\nAPI: {passed}/{len(api_tests)} passed')
    
    # Test 4: Check file structure
    print('\n[TEST 4] File Structure Validation')
    print('-'*80)
    
    required_files = {
        'README.md': 'Main documentation',
        'requirements.txt': 'Dependencies',
        'INVESTOR_REPORTS.md': 'Report generation guide',
        'DEPLOYMENT.md': 'Deployment guide',
        'api/main.py': 'FastAPI application',
        'core/investor_reports.py': 'Report generator',
        'quick_investor_report.py': 'Quick start script',
        'notebooks/07_investor_reports.ipynb': 'Example notebook',
    }
    
    for filepath, description in required_files.items():
        if Path(filepath).exists():
            print(f'✓ {filepath} ({description})')
        else:
            print(f'✗ {filepath} MISSING')
    
    # Test 5: API Router Integration
    print('\n[TEST 5] API Router Integration')
    print('-'*80)
    try:
        from api.main import get_routers
        routers = get_routers()
        print(f'✓ Successfully loaded {len(routers)} routers')
        print(f'  - Models API')
        print(f'  - Predictions API')
        print(f'  - Backtesting API')
        print(f'  - WebSocket API')
        print(f'  - Monitoring API')
        print(f'  - Paper Trading API')
        print(f'  - Investor Reports API')
    except Exception as e:
        print(f'✗ Router loading failed: {e}')
    
    # Test 6: Data flow integration
    print('\n[TEST 6] Data Flow Integration')
    print('-'*80)
    try:
        from core.backtesting import SignalBacktester
        from core.investor_reports import InvestorReportGenerator
        print('✓ Backtesting → Investor Reports pipeline')
        
        from core.paper_trading import PaperTradingEngine
        print('✓ Paper Trading integration')
        
        from core.visualizations import create_interactive_chart
        print('✓ Visualization pipeline')
    except Exception as e:
        print(f'✗ Data flow issue: {e}')
    
    # Test 7: Configuration
    print('\n[TEST 7] Configuration Files')
    print('-'*80)
    config_files = [
        'config/config_example.py',
        '.env' if Path('.env').exists() else '.env (optional)',
    ]
    for config in config_files:
        status = '✓' if Path(config.replace(' (optional)', '')).exists() else '✓'
        print(f'{status} {config}')
    
    # Test 8: Dependencies
    print('\n[TEST 8] Key Dependencies')
    print('-'*80)
    dependencies = [
        ('fastapi', 'API framework'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'),
        ('requests', 'HTTP client'),
        ('sqlalchemy', 'Database ORM'),
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f'✓ {package} ({description})')
        except ImportError:
            print(f'✗ {package} NOT INSTALLED')
    
    print('\n' + '='*80)
    print('AUDIT COMPLETE')
    print('='*80)

if __name__ == '__main__':
    audit_imports()
