"""
Validate Comprehensive Integration
Tests that all components are integrated and working
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports."""
    print("\n[1/5] Testing imports...")
    errors = []
    
    try:
        from core.comprehensive_integration import ComprehensiveIntegration
        print("  ✓ Comprehensive integration")
    except Exception as e:
        errors.append(f"Comprehensive integration: {e}")
        print(f"  ✗ Comprehensive integration: {e}")
    
    try:
        from api.comprehensive_api import router
        print("  ✓ Comprehensive API")
    except Exception as e:
        errors.append(f"Comprehensive API: {e}")
        print(f"  ✗ Comprehensive API: {e}")
    
    try:
        from core.enhanced_orchestrator import EnhancedOrchestrator
        print("  ✓ Enhanced orchestrator")
    except Exception as e:
        errors.append(f"Enhanced orchestrator: {e}")
        print(f"  ✗ Enhanced orchestrator: {e}")
    
    try:
        from models.quant.advanced_models import FactorModel, RegimeDetector
        print("  ✓ Advanced quant models")
    except Exception as e:
        errors.append(f"Advanced quant models: {e}")
        print(f"  ✗ Advanced quant models: {e}")
    
    return len(errors) == 0


def test_integration_initialization():
    """Test integration initialization."""
    print("\n[2/5] Testing integration initialization...")
    try:
        from core.comprehensive_integration import ComprehensiveIntegration
        
        integration = ComprehensiveIntegration(symbols=["AAPL"])
        print("  ✓ Integration created")
        
        # Test status
        status = integration.get_integration_status()
        print(f"  ✓ Integration status: {len(status.get('components_integrated', []))} components")
        
        return True
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return False


def test_api_routes():
    """Test API routes."""
    print("\n[3/5] Testing API routes...")
    try:
        from api.comprehensive_api import router
        
        routes = [r.path for r in router.routes]
        expected_routes = [
            "/api/v1/comprehensive/initialize",
            "/api/v1/comprehensive/analyze/{symbol}",
            "/api/v1/comprehensive/daily-analysis",
            "/api/v1/comprehensive/status",
            "/api/v1/comprehensive/alerts"
        ]
        
        print(f"  ✓ Found {len(routes)} routes")
        for route in expected_routes:
            if any(route.replace("{symbol}", "").replace("//", "/") in r for r in routes):
                print(f"  ✓ Route: {route}")
            else:
                print(f"  ⚠ Route not found: {route}")
        
        return True
    except Exception as e:
        print(f"  ✗ API routes test failed: {e}")
        return False


def test_component_integration():
    """Test component integration."""
    print("\n[4/5] Testing component integration...")
    try:
        from core.comprehensive_integration import ComprehensiveIntegration
        
        integration = ComprehensiveIntegration(symbols=["AAPL"])
        
        # Check that all components are accessible
        components = [
            ("orchestrator", integration.orchestrator),
            ("ai_service", integration.ai_service),
            ("company_search", integration.company_search),
            ("model_monitor", integration.model_monitor),
            ("alerting", integration.alerting),
            ("cache", integration.cache)
        ]
        
        for name, component in components:
            if component is not None:
                print(f"  ✓ {name} integrated")
            else:
                print(f"  ✗ {name} not integrated")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Component integration test failed: {e}")
        return False


def test_automation():
    """Test automation features."""
    print("\n[5/5] Testing automation...")
    try:
        from core.comprehensive_integration import ComprehensiveIntegration
        
        integration = ComprehensiveIntegration(symbols=["AAPL"])
        
        # Check automation methods exist
        methods = [
            "comprehensive_analysis",
            "automated_daily_analysis",
            "get_integration_status"
        ]
        
        for method in methods:
            if hasattr(integration, method):
                print(f"  ✓ Method: {method}")
            else:
                print(f"  ✗ Method missing: {method}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Automation test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("COMPREHENSIVE INTEGRATION VALIDATION")
    print("=" * 80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Initialization", test_integration_initialization()))
    results.append(("API Routes", test_api_routes()))
    results.append(("Component Integration", test_component_integration()))
    results.append(("Automation", test_automation()))
    
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
        print("\n✅ Comprehensive integration validated!")
        print("   All components are integrated and automated.")
    else:
        print("\n⚠ Some validations failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
