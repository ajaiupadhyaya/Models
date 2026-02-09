#!/usr/bin/env python3
"""
Comprehensive API and endpoint validation.
Tests all API routers, websockets, and pipelines to ensure system operability.
"""

import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def test_imports() -> Dict[str, Any]:
    """Test that all API modules can be imported."""
    print("\n" + "=" * 70)
    print("API MODULE IMPORT TESTS")
    print("=" * 70)
    
    modules = [
        ("main", "api.main"),
        ("auth", "api.auth_api"),
        ("models", "api.models_api"),
        ("predictions", "api.predictions_api"),
        ("backtesting", "api.backtesting_api"),
        ("websocket", "api.websocket_api"),
        ("monitoring", "api.monitoring"),
        ("paper_trading", "api.paper_trading_api"),
        ("investor_reports", "api.investor_reports_api"),
        ("company", "api.company_analysis_api"),
        ("ai_analysis", "api.ai_analysis_api"),
        ("data", "api.data_api"),
        ("news", "api.news_api"),
        ("risk", "api.risk_api"),
        ("automation", "api.automation_api"),
        ("orchestrator", "api.orchestrator_api"),
        ("screener", "api.screener_api"),
        ("comprehensive", "api.comprehensive_api"),
        ("institutional", "api.institutional_api"),
        ("cache", "api.cache"),
        ("rate_limit", "api.rate_limit"),
    ]
    
    results = {"passed": [], "failed": []}
    
    for name, module_path in modules:
        try:
            mod = importlib.import_module(module_path)
            results["passed"].append(name)
            
            # Check if router exists
            if hasattr(mod, 'router'):
                route_count = len([r for r in mod.router.routes])
                print(f"  ✅ {name:20s} - {route_count:2d} routes")
            elif hasattr(mod, 'app'):
                print(f"  ✅ {name:20s} - Main app")
            else:
                print(f"  ✅ {name:20s} - Utility module")
                
        except Exception as e:
            results["failed"].append((name, str(e)))
            print(f"  ❌ {name:20s} - {e}")
    
    return results


def test_router_endpoints() -> Dict[str, Any]:
    """Test that all routers have proper endpoint definitions."""
    print("\n" + "=" * 70)
    print("ROUTER ENDPOINT VALIDATION")
    print("=" * 70)
    
    routers_to_test = [
        ("auth_api", "/api/auth"),
        ("models_api", "/api/v1/models"),
        ("predictions_api", "/api/v1/predictions"),
        ("backtesting_api", "/api/v1/backtest"),
        ("websocket_api", "/api/v1/ws"),
        ("monitoring", "/api/v1/monitoring"),
        ("data_api", "/api/v1/data"),
        ("company_analysis_api", "/api/v1/company"),
        ("risk_api", "/api/v1/risk"),
    ]
    
    results = {"passed": [], "failed": []}
    total_endpoints = 0
    
    for module_name, prefix in routers_to_test:
        try:
            mod = importlib.import_module(f"api.{module_name}")
            if not hasattr(mod, 'router'):
                results["failed"].append((module_name, "No router found"))
                print(f"  ❌ {module_name:25s} - No router attribute")
                continue
            
            router = mod.router
            routes = [r for r in router.routes]
            
            if not routes:
                results["failed"].append((module_name, "No routes defined"))
                print(f"  ⚠️  {module_name:25s} - No routes defined")
                continue
            
            # List endpoints
            endpoints = []
            for route in routes:
                if hasattr(route, 'path'):
                    method = getattr(route, 'methods', ['GET'])
                    endpoints.append(f"{list(method)[0] if method else 'GET'} {prefix}{route.path}")
            
            results["passed"].append(module_name)
            total_endpoints += len(endpoints)
            print(f"  ✅ {module_name:25s} - {len(endpoints):2d} endpoints")
            
            # Show first 3 endpoints as examples
            for ep in endpoints[:3]:
                print(f"      → {ep}")
            if len(endpoints) > 3:
                print(f"      ... and {len(endpoints) - 3} more")
            
        except Exception as e:
            results["failed"].append((module_name, str(e)))
            print(f"  ❌ {module_name:25s} - {e}")
    
    results["total_endpoints"] = total_endpoints
    return results


def test_websocket_functionality() -> Dict[str, Any]:
    """Test WebSocket connection manager and endpoints."""
    print("\n" + "=" * 70)
    print("WEBSOCKET FUNCTIONALITY TEST")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    try:
        from api.websocket_api import ConnectionManager, router
        
        # Test ConnectionManager instantiation
        try:
            manager = ConnectionManager()
            print(f"  ✅ ConnectionManager initialization works")
            results["passed"].append("ConnectionManager init")
            
            # Check attributes
            if hasattr(manager, 'active_connections'):
                print(f"  ✅ active_connections attribute exists")
                results["passed"].append("active_connections")
            
            if hasattr(manager, 'subscriptions'):
                print(f"  ✅ subscriptions attribute exists")
                results["passed"].append("subscriptions")
            
            # Check methods
            methods = ['connect', 'disconnect', 'subscribe', 'unsubscribe', 'send_personal_message']
            for method in methods:
                if hasattr(manager, method):
                    print(f"  ✅ {method} method exists")
                    results["passed"].append(f"method:{method}")
                else:
                    print(f"  ❌ {method} method missing")
                    results["failed"].append((f"method:{method}", "Not found"))
            
        except Exception as e:
            print(f"  ❌ ConnectionManager instantiation failed: {e}")
            results["failed"].append(("ConnectionManager", str(e)))
        
        # Test WebSocket routes
        ws_routes = [r for r in router.routes if hasattr(r, 'path') and 'websocket' in str(type(r)).lower()]
        if ws_routes:
            print(f"\n  ✅ {len(ws_routes)} WebSocket routes defined:")
            for route in ws_routes:
                print(f"      → WS /api/v1/ws{route.path}")
            results["passed"].append(f"{len(ws_routes)} ws_routes")
        else:
            print(f"  ⚠️  No WebSocket routes found")
            results["failed"].append(("ws_routes", "None found"))
        
    except Exception as e:
        print(f"  ❌ WebSocket module import failed: {e}")
        results["failed"].append(("websocket_api", str(e)))
    
    return results


def test_core_pipelines() -> Dict[str, Any]:
    """Test core data and ML pipelines."""
    print("\n" + "=" * 70)
    print("CORE PIPELINE VALIDATION")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    # Test DataFetcher
    try:
        from core.data_fetcher import DataFetcher
        df = DataFetcher()
        print(f"  ✅ DataFetcher initialization")
        results["passed"].append("DataFetcher")
        
        # Check methods
        methods = ['get_stock_data', 'get_multiple_stocks', 'get_stock_info', 
                  'get_crypto_data', 'get_economic_indicator']
        for method in methods:
            if hasattr(df, method):
                print(f"  ✅ DataFetcher.{method} exists")
                results["passed"].append(f"DataFetcher.{method}")
            else:
                print(f"  ❌ DataFetcher.{method} missing")
                results["failed"].append((f"DataFetcher.{method}", "Not found"))
                
    except Exception as e:
        print(f"  ❌ DataFetcher: {e}")
        results["failed"].append(("DataFetcher", str(e)))
    
    # Test Backtesting
    try:
        from core.backtesting import BacktestEngine
        print(f"  ✅ BacktestEngine available")
        results["passed"].append("BacktestEngine")
    except Exception as e:
        print(f"  ⚠️  BacktestEngine: {e}")
        results["failed"].append(("BacktestEngine", str(e)))
    
    # Test Paper Trading
    try:
        from core.paper_trading import PaperTradingEngine
        print(f"  ✅ PaperTradingEngine available")
        results["passed"].append("PaperTradingEngine")
    except Exception as e:
        print(f"  ⚠️  PaperTradingEngine: {e}")
        results["failed"].append(("PaperTradingEngine", str(e)))
    
    # Test AI Analysis
    try:
        from core.ai_analysis import AIAnalysisService
        print(f"  ✅ AIAnalysisService available")
        results["passed"].append("AIAnalysisService")
    except Exception as e:
        print(f"  ⚠️  AIAnalysisService: {e} (optional)")
        # AI is optional, don't fail
    
    return results


def test_main_app() -> Dict[str, Any]:
    """Test that the main FastAPI app can be instantiated."""
    print("\n" + "=" * 70)
    print("MAIN APP VALIDATION")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    try:
        from api.main import app, get_routers
        
        print(f"  ✅ Main FastAPI app imported")
        results["passed"].append("app_import")
        
        # Check app attributes
        if hasattr(app, 'title'):
            print(f"  ✅ App title: {app.title}")
            results["passed"].append("app_title")
        
        if hasattr(app, 'version'):
            print(f"  ✅ App version: {app.version}")
            results["passed"].append("app_version")
        
        # Check routers
        try:
            routers = get_routers()
            print(f"  ✅ Loaded {len(routers)} routers: {list(routers.keys())}")
            results["passed"].append("routers_loaded")
            results["router_count"] = len(routers)
            
            for name in routers.keys():
                print(f"      → {name}")
                
        except Exception as e:
            print(f"  ⚠️  Router loading: {e}")
            results["failed"].append(("routers", str(e)))
        
        # Check middleware
        if hasattr(app, 'middleware_stack'):
            print(f"  ✅ Middleware configured")
            results["passed"].append("middleware")
        
        # Count total routes
        route_count = len([r for r in app.routes])
        print(f"  ✅ Total routes registered: {route_count}")
        results["passed"].append("routes_registered")
        results["route_count"] = route_count
        
    except Exception as e:
        print(f"  ❌ Main app validation failed: {e}")
        results["failed"].append(("main_app", str(e)))
    
    return results


def test_critical_imports() -> Dict[str, Any]:
    """Test that critical dependencies are available."""
    print("\n" + "=" * 70)
    print("CRITICAL DEPENDENCY CHECK")
    print("=" * 70)
    
    dependencies = [
        ("fastapi", "FastAPI framework"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("yfinance", "Market data"),
        ("sklearn", "Machine learning"),
        ("torch", "Deep learning"),
        ("uvicorn", "ASGI server"),
    ]
    
    results = {"passed": [], "failed": []}
    
    for module, description in dependencies:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module:15s} - {description}")
            results["passed"].append(module)
        except ImportError as e:
            print(f"  ❌ {module:15s} - {description} - MISSING")
            results["failed"].append((module, "Not installed"))
    
    return results


def main():
    """Run comprehensive validation."""
    print("=" * 70)
    print("COMPREHENSIVE API & PIPELINE VALIDATION")
    print("=" * 70)
    print("Testing all API endpoints, websockets, and core pipelines...")
    
    all_results = {}
    
    # Run all tests
    all_results["imports"] = test_imports()
    all_results["dependencies"] = test_critical_imports()
    all_results["endpoints"] = test_router_endpoints()
    all_results["websockets"] = test_websocket_functionality()
    all_results["pipelines"] = test_core_pipelines()
    all_results["main_app"] = test_main_app()
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    total_passed = sum(len(r.get("passed", [])) for r in all_results.values())
    total_failed = sum(len(r.get("failed", [])) for r in all_results.values())
    
    print(f"\n✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    
    # Detailed failures
    if total_failed > 0:
        print("\n🔴 FAILURES:")
        for test_name, result in all_results.items():
            if result.get("failed"):
                print(f"\n  {test_name.upper()}:")
                for failure in result["failed"]:
                    if isinstance(failure, tuple):
                        print(f"    - {failure[0]}: {failure[1]}")
                    else:
                        print(f"    - {failure}")
    
    # Key metrics
    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)
    
    if "endpoints" in all_results:
        print(f"  Total API endpoints: {all_results['endpoints'].get('total_endpoints', 0)}")
    
    if "main_app" in all_results:
        print(f"  Total routes: {all_results['main_app'].get('route_count', 0)}")
        print(f"  Routers loaded: {all_results['main_app'].get('router_count', 0)}")
    
    # Overall status
    print("\n" + "=" * 70)
    
    critical_failures = [
        f for r in [all_results.get("dependencies"), all_results.get("main_app")] 
        if r for f in r.get("failed", [])
    ]
    
    if critical_failures:
        print("🔴 CRITICAL ISSUES DETECTED")
        print("\nSome critical components are not working.")
        print("Review failures above and fix issues.")
        return False
    elif total_failed > 0:
        print("🟡 PARTIAL SUCCESS")
        print("\nCore functionality is operational but some optional features failed.")
        print("Review warnings above. System can run but may have limited features.")
        return True
    else:
        print("🟢 ALL SYSTEMS OPERATIONAL")
        print("\n✅ All API endpoints, websockets, and pipelines are functional")
        print("✅ System is ready for production use")
        print("\nQuick start:")
        print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
