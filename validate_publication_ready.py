"""
Comprehensive Validation Script for Publication Readiness
Validates all components, dependencies, and functionality
"""

import sys
from pathlib import Path
import importlib
import logging
from typing import Dict, List, Tuple
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class PublicationValidator:
    """Validate project readiness for publication."""
    
    def __init__(self):
        """Initialize validator."""
        self.results = {
            'dependencies': {},
            'modules': {},
            'apis': {},
            'models': {},
            'visualizations': {},
            'automation': {},
            'overall': {}
        }
        self.errors = []
        self.warnings = []
    
    def validate_dependencies(self) -> Dict:
        """Validate all required dependencies."""
        logger.info("Validating dependencies...")
        
        required_deps = {
            'Core': ['numpy', 'pandas', 'scipy', 'scikit-learn'],
            'Financial': ['yfinance', 'fredapi', 'alpha_vantage'],
            'Visualization': ['plotly', 'dash', 'matplotlib', 'seaborn'],
            'ML/DL': ['sklearn', 'tensorflow', 'keras'],
            'RL': ['stable_baselines3', 'gym'],
            'Transformers': ['transformers', 'torch'],
            'API': ['fastapi', 'uvicorn', 'pydantic'],
            'D3.js Bridge': ['js2py']
        }
        
        results = {}
        for category, deps in required_deps.items():
            results[category] = {}
            for dep in deps:
                try:
                    mod = importlib.import_module(dep.replace('-', '_'))
                    version = getattr(mod, '__version__', 'unknown')
                    results[category][dep] = {'status': '✓', 'version': version}
                except ImportError:
                    results[category][dep] = {'status': '✗', 'version': None}
                    self.warnings.append(f"Missing dependency: {dep}")
        
        self.results['dependencies'] = results
        return results
    
    def validate_core_modules(self) -> Dict:
        """Validate core modules."""
        logger.info("Validating core modules...")
        
        core_modules = {
            'data_fetcher': 'core.data_fetcher',
            'backtesting': 'core.backtesting',
            'paper_trading': 'core.paper_trading',
            'investor_reports': 'core.investor_reports',
            'visualizations': 'core.visualizations',
            'utils': 'core.utils',
            'data_cache': 'core.data_cache',
            'dashboard': 'core.dashboard'
        }
        
        results = {}
        for name, module_path in core_modules.items():
            try:
                mod = importlib.import_module(module_path)
                results[name] = {'status': '✓', 'classes': dir(mod)}
            except Exception as e:
                results[name] = {'status': '✗', 'error': str(e)}
                self.errors.append(f"Core module {name} failed: {e}")
        
        self.results['modules'] = results
        return results
    
    def validate_models(self) -> Dict:
        """Validate financial models."""
        logger.info("Validating financial models...")
        
        model_categories = {
            'Valuation': ['models.valuation.dcf_model'],
            'Options': ['models.options.black_scholes'],
            'Portfolio': ['models.portfolio.optimization'],
            'Risk': ['models.risk.var_cvar', 'models.risk.stress_testing'],
            'Trading': ['models.trading.strategies', 'models.trading.backtesting'],
            'Macro': ['models.macro.macro_indicators', 'models.macro.geopolitical_risk'],
            'ML': ['models.ml.forecasting', 'models.ml.advanced_trading', 'models.ml.transformer_models'],
            'Sentiment': ['models.sentiment.market_sentiment'],
            'Fixed Income': ['models.fixed_income.bond_analytics']
        }
        
        results = {}
        for category, modules in model_categories.items():
            results[category] = {}
            for module_path in modules:
                try:
                    mod = importlib.import_module(module_path)
                    results[category][module_path] = {'status': '✓'}
                except Exception as e:
                    results[category][module_path] = {'status': '✗', 'error': str(e)}
                    self.warnings.append(f"Model module {module_path} failed: {e}")
        
        self.results['models'] = results
        return results
    
    def validate_apis(self) -> Dict:
        """Validate API endpoints."""
        logger.info("Validating API endpoints...")
        
        api_modules = {
            'models_api': 'api.models_api',
            'predictions_api': 'api.predictions_api',
            'backtesting_api': 'api.backtesting_api',
            'websocket_api': 'api.websocket_api',
            'monitoring': 'api.monitoring',
            'paper_trading_api': 'api.paper_trading_api',
            'investor_reports_api': 'api.investor_reports_api',
            'main': 'api.main'
        }
        
        results = {}
        for name, module_path in api_modules.items():
            try:
                mod = importlib.import_module(module_path)
                # Check for router
                if hasattr(mod, 'router'):
                    results[name] = {'status': '✓', 'has_router': True}
                else:
                    results[name] = {'status': '✓', 'has_router': False}
            except Exception as e:
                results[name] = {'status': '✗', 'error': str(e)}
                self.errors.append(f"API module {name} failed: {e}")
        
        self.results['apis'] = results
        return results
    
    def validate_visualizations(self) -> Dict:
        """Validate visualization modules."""
        logger.info("Validating visualizations...")
        
        viz_modules = {
            'Plotly Charts': 'core.visualizations',
            'Interactive Charts': 'core.advanced_viz.interactive_charts',
            'D3 Visualizations': 'core.advanced_viz.d3_visualizations',
            'Market Analysis Viz': 'core.advanced_viz.market_analysis_viz',
            'Portfolio Viz': 'core.advanced_viz.portfolio_visualizations'
        }
        
        results = {}
        for name, module_path in viz_modules.items():
            try:
                mod = importlib.import_module(module_path)
                results[name] = {'status': '✓'}
            except Exception as e:
                results[name] = {'status': '✗', 'error': str(e)}
                if 'd3' in name.lower():
                    self.warnings.append(f"Visualization {name} failed (optional): {e}")
                else:
                    self.errors.append(f"Visualization {name} failed: {e}")
        
        self.results['visualizations'] = results
        return results
    
    def validate_automation(self) -> Dict:
        """Validate automation components."""
        logger.info("Validating automation...")
        
        automation_modules = {
            'Orchestrator': 'automation.orchestrator',
            'ML Pipeline': 'automation.ml_pipeline',
            'Data Pipeline': 'automation.data_pipeline',
            'Trading Automation': 'automation.trading_automation'
        }
        
        results = {}
        for name, module_path in automation_modules.items():
            try:
                mod = importlib.import_module(module_path)
                results[name] = {'status': '✓'}
            except Exception as e:
                results[name] = {'status': '✗', 'error': str(e)}
                self.warnings.append(f"Automation {name} failed: {e}")
        
        self.results['automation'] = results
        return results
    
    def run_all_validations(self) -> Dict:
        """Run all validation checks."""
        logger.info("="*80)
        logger.info("PUBLICATION READINESS VALIDATION")
        logger.info("="*80)
        
        self.validate_dependencies()
        self.validate_core_modules()
        self.validate_models()
        self.validate_apis()
        self.validate_visualizations()
        self.validate_automation()
        
        # Overall assessment
        total_checks = sum(
            len(v) if isinstance(v, dict) else 1
            for result_dict in self.results.values()
            if isinstance(result_dict, dict)
            for v in result_dict.values()
        )
        
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        
        self.results['overall'] = {
            'total_checks': total_checks,
            'errors': error_count,
            'warnings': warning_count,
            'ready_for_publication': error_count == 0,
            'status': 'READY' if error_count == 0 else 'NEEDS FIXES'
        }
        
        return self.results
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        
        for category, results in self.results.items():
            if category == 'overall':
                continue
            
            print(f"\n{category.upper()}:")
            print("-" * 80)
            
            if isinstance(results, dict):
                for name, status in results.items():
                    if isinstance(status, dict):
                        status_str = status.get('status', '?')
                        print(f"  {status_str} {name}")
                        if 'error' in status:
                            print(f"      Error: {status['error']}")
        
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        overall = self.results['overall']
        print(f"Total Checks: {overall['total_checks']}")
        print(f"Errors: {overall['errors']}")
        print(f"Warnings: {overall['warnings']}")
        print(f"Status: {overall['status']}")
        print(f"Ready for Publication: {'YES ✓' if overall['ready_for_publication'] else 'NO ✗'}")
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    validator = PublicationValidator()
    results = validator.run_all_validations()
    validator.print_report()
    
    # Exit with error code if not ready
    if not results['overall']['ready_for_publication']:
        sys.exit(1)
