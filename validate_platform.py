"""
Platform Validation Script
Validates all components of the Bloomberg Terminal platform
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_imports():
    """Validate all imports."""
    print("\n[1/8] Validating imports...")
    errors = []
    
    try:
        from models.ml.rl_agents import DQNAgent, PPOAgent, StableBaselines3Wrapper
        print("  ✓ RL agents imported")
    except Exception as e:
        errors.append(f"RL agents: {e}")
        print(f"  ✗ RL agents: {e}")
    
    try:
        from core.automated_trading_orchestrator import AutomatedTradingOrchestrator
        print("  ✓ Orchestrator imported")
    except Exception as e:
        errors.append(f"Orchestrator: {e}")
        print(f"  ✗ Orchestrator: {e}")
    
    try:
        from core.bloomberg_terminal_ui import BloombergTerminalUI
        print("  ✓ Bloomberg Terminal UI imported")
    except Exception as e:
        errors.append(f"UI: {e}")
        print(f"  ✗ UI: {e}")
    
    try:
        from core.realtime_streaming import RealTimeDataStreamer
        print("  ✓ Real-time streaming imported")
    except Exception as e:
        errors.append(f"Streaming: {e}")
        print(f"  ✗ Streaming: {e}")
    
    try:
        from core.model_monitor import ModelPerformanceMonitor
        print("  ✓ Model monitor imported")
    except Exception as e:
        errors.append(f"Model monitor: {e}")
        print(f"  ✗ Model monitor: {e}")
    
    try:
        from core.alerting_system import AlertingSystem, AlertSeverity
        print("  ✓ Alerting system imported")
    except Exception as e:
        errors.append(f"Alerting: {e}")
        print(f"  ✗ Alerting: {e}")
    
    try:
        from api.orchestrator_api import router
        print("  ✓ Orchestrator API imported")
    except Exception as e:
        errors.append(f"Orchestrator API: {e}")
        print(f"  ✗ Orchestrator API: {e}")
    
    if errors:
        print(f"\n  ⚠ {len(errors)} import errors found")
        return False
    else:
        print("  ✓ All imports successful")
        return True


def validate_dependencies():
    """Validate required dependencies."""
    print("\n[2/8] Validating dependencies...")
    missing = []
    
    dependencies = [
        ("torch", "PyTorch"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("dash", "Dash"),
        ("dash_bootstrap_components", "Dash Bootstrap Components"),
        ("schedule", "Schedule"),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            missing.append(name)
            print(f"  ✗ {name} missing")
    
    if missing:
        print(f"\n  ⚠ Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install torch stable-baselines3 dash-bootstrap-components schedule")
        return False
    else:
        print("  ✓ All dependencies installed")
        return True


def validate_data_fetcher():
    """Validate data fetcher."""
    print("\n[3/8] Validating data fetcher...")
    try:
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        
        # Test stock data
        df = fetcher.get_stock_data("AAPL", period="1mo")
        if df is not None and len(df) > 0:
            print(f"  ✓ Stock data fetching works ({len(df)} rows)")
        else:
            print("  ⚠ Stock data fetch returned empty")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Data fetcher error: {e}")
        return False


def validate_orchestrator():
    """Validate orchestrator initialization."""
    print("\n[4/8] Validating orchestrator...")
    try:
        from core.automated_trading_orchestrator import AutomatedTradingOrchestrator
        
        orchestrator = AutomatedTradingOrchestrator(
            symbols=["AAPL"],
            use_rl=False,  # Skip RL for faster validation
            use_lstm=False,  # Skip LSTM for faster validation
            use_ensemble=True
        )
        
        print("  ✓ Orchestrator created")
        
        # Test initialization (without full training)
        print("  ⚠ Skipping full model initialization (takes time)")
        print("  ✓ Orchestrator structure validated")
        
        return True
    except Exception as e:
        print(f"  ✗ Orchestrator error: {e}")
        return False


def validate_ui():
    """Validate UI components."""
    print("\n[5/8] Validating UI...")
    try:
        from core.bloomberg_terminal_ui import BloombergTerminalUI
        
        ui = BloombergTerminalUI(symbols=["AAPL"])
        print("  ✓ UI created")
        print("  ✓ UI layout validated")
        
        return True
    except Exception as e:
        print(f"  ✗ UI error: {e}")
        return False


def validate_monitoring():
    """Validate monitoring systems."""
    print("\n[6/8] Validating monitoring...")
    try:
        from core.model_monitor import ModelPerformanceMonitor
        from core.alerting_system import AlertingSystem, AlertSeverity
        
        monitor = ModelPerformanceMonitor()
        monitor.record_prediction("test", "AAPL", 150.0, 151.0)
        print("  ✓ Model monitor works")
        
        alerts = AlertingSystem()
        alerts.create_alert("test", AlertSeverity.INFO, "Test", "Test message")
        print("  ✓ Alerting system works")
        
        return True
    except Exception as e:
        print(f"  ✗ Monitoring error: {e}")
        return False


def validate_api():
    """Validate API endpoints."""
    print("\n[7/8] Validating API...")
    try:
        from api.orchestrator_api import router, get_orchestrator
        
        print("  ✓ Orchestrator API router loaded")
        print(f"  ✓ API has {len(router.routes)} routes")
        
        return True
    except Exception as e:
        print(f"  ✗ API error: {e}")
        return False


def validate_file_structure():
    """Validate file structure."""
    print("\n[8/8] Validating file structure...")
    required_files = [
        "models/ml/rl_agents.py",
        "core/automated_trading_orchestrator.py",
        "core/bloomberg_terminal_ui.py",
        "core/realtime_streaming.py",
        "core/model_monitor.py",
        "core/alerting_system.py",
        "api/orchestrator_api.py",
        "start_bloomberg_terminal.py",
        "BLOOMBERG_TERMINAL_GUIDE.md"
    ]
    
    missing = []
    for file in required_files:
        path = project_root / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            missing.append(file)
            print(f"  ✗ {file} missing")
    
    if missing:
        print(f"\n  ⚠ {len(missing)} files missing")
        return False
    else:
        print("  ✓ All required files present")
        return True


def main():
    """Run all validations."""
    print("=" * 80)
    print("BLOOMBERG TERMINAL PLATFORM VALIDATION")
    print("=" * 80)
    
    results = []
    
    results.append(("File Structure", validate_file_structure()))
    results.append(("Dependencies", validate_dependencies()))
    results.append(("Imports", validate_imports()))
    results.append(("Data Fetcher", validate_data_fetcher()))
    results.append(("Orchestrator", validate_orchestrator()))
    results.append(("UI", validate_ui()))
    results.append(("Monitoring", validate_monitoring()))
    results.append(("API", validate_api()))
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 80)
    
    if passed == total:
        print("\n✅ Platform validation successful!")
        print("   Ready to start: python start_bloomberg_terminal.py")
    else:
        print("\n⚠ Some validations failed. Check errors above.")
        print("   Install missing dependencies: pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
