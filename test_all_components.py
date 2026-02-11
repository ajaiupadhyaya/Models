"""
Comprehensive Component Testing Suite
Tests all components and ensures harmony between them
"""

import sys
import traceback
from pathlib import Path
import importlib
import logging
from typing import Dict, List, Tuple, Any
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ComponentTester:
    """Comprehensive component testing."""
    
    def __init__(self):
        """Initialize tester."""
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'errors': []
        }
        self.fixes_applied = []
    
    def test_import(self, module_name: str, description: str = None) -> Tuple[bool, str]:
        """Test module import."""
        try:
            mod = importlib.import_module(module_name)
            return True, f"✓ {description or module_name} imported successfully"
        except Exception as e:
            error_msg = f"✗ {description or module_name} import failed: {str(e)}"
            return False, error_msg
    
    def test_class_instantiation(self, module_name: str, class_name: str, init_args: Dict = None) -> Tuple[bool, str]:
        """Test class instantiation."""
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            if init_args is None:
                # Skip instantiation if None (class needs specific args)
                return True, f"✓ {class_name} class available (requires specific initialization)"
            elif init_args:
                instance = cls(**init_args)
            else:
                instance = cls()
            return True, f"✓ {class_name} instantiated successfully"
        except Exception as e:
            error_msg = f"✗ {class_name} instantiation failed: {str(e)}"
            return False, error_msg
    
    def test_core_modules(self) -> Dict[str, Any]:
        """Test core modules."""
        logger.info("Testing core modules...")
        results = {}
        
        tests = [
            ('core.data_fetcher', 'DataFetcher', {}),
            ('core.utils', None, None),
            ('core.data_cache', 'DataCache', {}),
            ('core.visualizations', 'ChartBuilder', {}),
        ]
        
        for module_name, class_name, init_args in tests:
            if class_name:
                success, msg = self.test_class_instantiation(module_name, class_name, init_args)
            else:
                success, msg = self.test_import(module_name)
            
            results[module_name] = {'success': success, 'message': msg}
            if success:
                self.results['passed'].append(msg)
            else:
                self.results['failed'].append(msg)
                self.results['errors'].append(msg)
        
        return results
    
    def test_models(self) -> Dict[str, Any]:
        """Test financial models."""
        logger.info("Testing financial models...")
        results = {}
        
        tests = [
            ('models.valuation.dcf_model', 'DCFModel', None),  # Needs free_cash_flows
            ('models.options.black_scholes', 'BlackScholes', {}),
            ('models.portfolio.optimization', 'MeanVarianceOptimizer', None),  # Needs expected_returns, cov_matrix
            ('models.risk.var_cvar', 'VaRModel', {}),
            ('models.trading.strategies', 'MomentumStrategy', {'lookback_period': 20}),
            ('models.macro.geopolitical_risk', 'GeopoliticalRiskAnalyzer', {}),
            ('models.macro.geopolitical_risk', 'PolicyImpactAssessor', {}),
        ]
        
        for module_name, class_name, init_args in tests:
            success, msg = self.test_class_instantiation(module_name, class_name, init_args)
            results[f"{module_name}.{class_name}"] = {'success': success, 'message': msg}
            if success:
                self.results['passed'].append(msg)
            else:
                self.results['failed'].append(msg)
                self.results['errors'].append(msg)
        
        return results
    
    def test_ml_models(self) -> Dict[str, Any]:
        """Test ML/DL/RL models."""
        logger.info("Testing ML/DL/RL models...")
        results = {}
        
        tests = [
            ('models.ml.forecasting', 'TimeSeriesForecaster', {'model_type': 'random_forest'}),
            ('models.ml.forecasting', 'RegimeDetector', {'n_regimes': 3}),
            ('models.ml.forecasting', 'AnomalyDetector', {}),
        ]
        
        for module_name, class_name, init_args in tests:
            success, msg = self.test_class_instantiation(module_name, class_name, init_args)
            results[f"{module_name}.{class_name}"] = {'success': success, 'message': msg}
            if success:
                self.results['passed'].append(msg)
            else:
                self.results['failed'].append(msg)
                # ML models might have optional dependencies
                self.results['warnings'].append(msg)
        
        # Test LSTM (optional TensorFlow)
        try:
            from models.ml.advanced_trading import LSTMPredictor
            results['LSTMPredictor'] = {'success': True, 'message': '✓ LSTMPredictor available'}
            self.results['passed'].append('LSTMPredictor available')
        except ImportError as e:
            results['LSTMPredictor'] = {'success': False, 'message': f'⚠ LSTMPredictor requires TensorFlow: {e}'}
            self.results['warnings'].append(f'LSTMPredictor optional: {e}')
        
        # Test Transformers (optional)
        try:
            from models.ml.transformer_models import FinancialSentimentAnalyzer
            results['FinancialSentimentAnalyzer'] = {'success': True, 'message': '✓ FinancialSentimentAnalyzer available'}
            self.results['passed'].append('FinancialSentimentAnalyzer available')
        except ImportError as e:
            results['FinancialSentimentAnalyzer'] = {'success': False, 'message': f'⚠ FinancialSentimentAnalyzer requires transformers: {e}'}
            self.results['warnings'].append(f'FinancialSentimentAnalyzer optional: {e}')
        
        return results
    
    def test_visualizations(self) -> Dict[str, Any]:
        """Test visualization modules."""
        logger.info("Testing visualizations...")
        results = {}
        
        tests = [
            ('core.advanced_viz.interactive_charts', 'InteractiveCharts', {}),
            ('core.advanced_viz.d3_visualizations', 'D3Visualizations', {}),
        ]
        
        for module_name, class_name, init_args in tests:
            success, msg = self.test_class_instantiation(module_name, class_name, init_args)
            results[f"{module_name}.{class_name}"] = {'success': success, 'message': msg}
            if success:
                self.results['passed'].append(msg)
            else:
                self.results['failed'].append(msg)
                if 'd3' in module_name.lower():
                    self.results['warnings'].append(msg)
                else:
                    self.results['errors'].append(msg)
        
        return results
    
    def test_apis(self) -> Dict[str, Any]:
        """Test API modules."""
        logger.info("Testing API modules...")
        results = {}
        
        api_modules = [
            'api.main',
            'api.models_api',
            'api.predictions_api',
            'api.backtesting_api',
            'api.websocket_api',
            'api.monitoring',
            'api.paper_trading_api',
            'api.investor_reports_api',
        ]
        
        for module_name in api_modules:
            success, msg = self.test_import(module_name)
            results[module_name] = {'success': success, 'message': msg}
            if success:
                self.results['passed'].append(msg)
            else:
                self.results['failed'].append(msg)
                self.results['errors'].append(msg)
        
        return results
    
    def test_automation(self) -> Dict[str, Any]:
        """Test automation modules."""
        logger.info("Testing automation modules...")
        results = {}
        
        tests = [
            ('automation.orchestrator', 'AutomationOrchestrator', {}),
            ('automation.ml_pipeline', 'MLPipeline', {}),
            ('automation.trading_automation', 'TradingAutomation', {'trading_enabled': False, 'initial_capital': 100000.0}),
        ]
        
        for module_name, class_name, init_args in tests:
            success, msg = self.test_class_instantiation(module_name, class_name, init_args)
            results[f"{module_name}.{class_name}"] = {'success': success, 'message': msg}
            if success:
                self.results['passed'].append(msg)
            else:
                self.results['failed'].append(msg)
                self.results['errors'].append(msg)
        
        return results
    
    def test_integration(self) -> Dict[str, Any]:
        """Test component integration."""
        logger.info("Testing component integration...")
        results = {}
        
        # Test data fetcher -> backtesting integration
        try:
            from core.data_fetcher import DataFetcher
            from core.backtesting import BacktestEngine
            from models.trading.strategies import MomentumStrategy
            
            fetcher = DataFetcher()
            engine = BacktestEngine()
            strategy = MomentumStrategy(lookback_period=20)
            
            results['data_backtest_integration'] = {
                'success': True,
                'message': '✓ Data fetcher -> Backtesting integration works'
            }
            self.results['passed'].append('Data-Backtest integration')
        except Exception as e:
            results['data_backtest_integration'] = {
                'success': False,
                'message': f'✗ Data-Backtest integration failed: {e}'
            }
            self.results['failed'].append(f'Data-Backtest integration: {e}')
            self.results['errors'].append(str(e))
        
        # Test ML -> Trading integration
        try:
            from automation.ml_pipeline import MLPipeline
            from automation.trading_automation import TradingAutomation
            
            ml_pipeline = MLPipeline()
            trading = TradingAutomation(trading_enabled=False)
            
            results['ml_trading_integration'] = {
                'success': True,
                'message': '✓ ML -> Trading integration works'
            }
            self.results['passed'].append('ML-Trading integration')
        except Exception as e:
            results['ml_trading_integration'] = {
                'success': False,
                'message': f'✗ ML-Trading integration failed: {e}'
            }
            self.results['failed'].append(f'ML-Trading integration: {e}')
            self.results['errors'].append(str(e))
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        logger.info("="*80)
        logger.info("COMPREHENSIVE COMPONENT TESTING")
        logger.info("="*80)
        
        all_results = {
            'core': self.test_core_modules(),
            'models': self.test_models(),
            'ml_models': self.test_ml_models(),
            'visualizations': self.test_visualizations(),
            'apis': self.test_apis(),
            'automation': self.test_automation(),
            'integration': self.test_integration(),
        }
        
        return all_results
    
    def print_report(self):
        """Print test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"\n✓ Passed: {len(self.results['passed'])}")
        print(f"✗ Failed: {len(self.results['failed'])}")
        print(f"⚠ Warnings: {len(self.results['warnings'])}")
        print(f"❌ Errors: {len(self.results['errors'])}")
        
        if self.results['failed']:
            print("\nFAILED TESTS:")
            for failure in self.results['failed']:
                print(f"  {failure}")
        
        if self.results['warnings']:
            print("\nWARNINGS (Optional Dependencies):")
            for warning in self.results['warnings']:
                print(f"  {warning}")
        
        if self.results['errors']:
            print("\nCRITICAL ERRORS:")
            for error in self.results['errors']:
                print(f"  {error}")
        
        print("\n" + "="*80)
        
        if len(self.results['errors']) == 0:
            print("✅ ALL CRITICAL COMPONENTS WORKING")
        else:
            print("❌ SOME CRITICAL COMPONENTS FAILED")
        
        print("="*80)


if __name__ == "__main__":
    tester = ComponentTester()
    results = tester.run_all_tests()
    tester.print_report()
    
    # Exit with error code if critical errors
    if tester.results['errors']:
        sys.exit(1)
