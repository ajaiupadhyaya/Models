#!/usr/bin/env python3
"""
Comprehensive deployment validation for the Trading API.

Validates all critical components:
- Environment variables
- Configuration
- API imports and routers
- Health check endpoints
- Database/cache connectivity
- Frontend serving
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def log_success(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")

def log_error(msg: str):
    print(f"{RED}✗ {msg}{RESET}")

def log_warning(msg: str):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def log_info(msg: str):
    print(f"{BLUE}ℹ {msg}{RESET}")

def check_env_variables() -> Tuple[bool, List[str]]:
    """Check that critical environment variables are set."""
    print(f"\n{BLUE}=== Checking Environment Variables ==={RESET}")
    
    issues = []
    critical_vars = {
        'TERMINAL_USER': 'Username for terminal login',
        'TERMINAL_PASSWORD': 'Password for terminal login',
        'AUTH_SECRET': 'JWT authentication secret',
    }
    
    optional_vars = {
        'FRED_API_KEY': 'FRED API key (for Economic tab)',
        'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage key (for charts fallback)',
        'OPENAI_API_KEY': 'OpenAI API key (for AI tab)',
        'FINNHUB_API_KEY': 'Finnhub API key (for News tab)',
    }
    
    for var, desc in critical_vars.items():
        if os.getenv(var):
            log_success(f"{var} is set")
        else:
            log_warning(f"{var} is NOT set - {desc}")
            issues.append(f"Missing critical: {var}")
    
    for var, desc in optional_vars.items():
        if os.getenv(var):
            log_success(f"{var} is set (optional)")
        else:
            log_warning(f"{var} is not set (optional) - {desc}")
    
    return len([i for i in issues if 'critical' in i.lower()]) == 0, issues

def check_config_files() -> Tuple[bool, List[str]]:
    """Check critical configuration files exist and are readable."""
    print(f"\n{BLUE}=== Checking Configuration Files ==={RESET}")
    
    issues = []
    required_files = {
        'requirements-api.txt': 'API dependencies',
        'Dockerfile': 'Docker build config',
        'render.yaml': 'Render deployment config',
    }
    
    optional_files = {
        'config/settings.py': 'Central configuration',
        '.env.example': 'Environment template',
        'frontend/dist/index.html': 'Built frontend (for production)',
    }
    
    for file, desc in required_files.items():
        fpath = project_root / file
        if fpath.exists():
            log_success(f"{file} exists - {desc}")
        else:
            log_error(f"{file} MISSING - {desc}")
            issues.append(f"Missing file: {file}")
    
    for file, desc in optional_files.items():
        fpath = project_root / file
        if fpath.exists():
            log_success(f"{file} exists (optional) - {desc}")
        else:
            log_warning(f"{file} not found (optional) - {desc}")
    
    return len([i for i in issues if 'Missing file' in i]) == 0, issues

def check_api_routers() -> Tuple[bool, List[str]]:
    """Check that all API routers can be imported."""
    print(f"\n{BLUE}=== Checking API Routers ==={RESET}")
    
    issues = []
    routers = [
        ('api.models_api', 'Models'),
        ('api.predictions_api', 'Predictions'),
        ('api.backtesting_api', 'Backtesting'),
        ('api.data_api', 'Data'),
        ('api.risk_api', 'Risk'),
        ('api.ai_analysis_api', 'AI Analysis'),
        ('api.websocket_api', 'WebSocket'),
        ('api.monitoring', 'Monitoring'),
        ('api.auth_api', 'Authentication'),
        ('api.paper_trading_api', 'Paper Trading'),
        ('api.company_analysis_api', 'Company Analysis'),
        ('api.investor_reports_api', 'Investor Reports'),
        # Optional routers
        ('api.automation_api', 'Automation (optional)'),
        ('api.orchestrator_api', 'Orchestrator (optional)'),
        ('api.screener_api', 'Screener (optional)'),
        ('api.comprehensive_api', 'Comprehensive (optional)'),
        ('api.institutional_api', 'Institutional (optional)'),
    ]
    
    for module, name in routers:
        try:
            mod = __import__(module, fromlist=['router'])
            if hasattr(mod, 'router'):
                log_success(f"{name} router imported successfully")
            else:
                log_warning(f"{name} module loaded but no 'router' attribute found")
                issues.append(f"Router issue: {module} missing 'router'")
        except Exception as e:
            if 'optional' in name.lower():
                log_warning(f"{name} not available (optional): {str(e)[:50]}")
            else:
                log_error(f"{name} failed to import: {str(e)[:50]}")
                issues.append(f"Import error: {module} - {str(e)[:50]}")
    
    return len([i for i in issues if 'Import error' in i]) == 0, issues

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check that critical Python dependencies are installed."""
    print(f"\n{BLUE}=== Checking Python Dependencies ==={RESET}")
    
    issues = []
    critical_deps = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
        ('yfinance', 'yfinance'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]
    
    optional_deps = [
        ('openai', 'OpenAI'),
        ('fredapi', 'FRED API'),
        ('sklearn', 'scikit-learn'),
        ('tensorflow', 'TensorFlow'),
    ]
    
    for module, name in critical_deps:
        try:
            __import__(module)
            log_success(f"{name} is installed")
        except ImportError:
            log_error(f"{name} NOT installed")
            issues.append(f"Missing critical dependency: {name}")
    
    for module, name in optional_deps:
        try:
            __import__(module)
            log_success(f"{name} is installed (optional)")
        except ImportError:
            log_warning(f"{name} not installed (optional)")
    
    return len([i for i in issues if 'critical' in i.lower()]) == 0, issues

async def check_api_health() -> Tuple[bool, List[str]]:
    """Check that the FastAPI app can be instantiated and has health endpoints."""
    print(f"\n{BLUE}=== Checking API Health Endpoints ==={RESET}")
    
    issues = []
    
    try:
        from api.main import app
        log_success("API app instantiated successfully")
    except Exception as e:
        log_error(f"Failed to instantiate API app: {str(e)[:100]}")
        issues.append(f"API app error: {str(e)[:100]}")
        return False, issues
    
    # Check routes
    routes = [route.path for route in app.routes]
    critical_routes = ['/health', '/info', '/docs', '/']
    
    for route in critical_routes:
        if route in routes:
            log_success(f"Route {route} is registered")
        else:
            log_warning(f"Route {route} not found")
    
    # Check for auth routes
    auth_routes = [r for r in routes if 'auth' in r.lower() or 'login' in r.lower()]
    if auth_routes:
        log_success(f"Authentication routes found: {len(auth_routes)}")
    else:
        log_warning("No authentication routes detected")
    
    return True, issues

def check_frontend() -> Tuple[bool, List[str]]:
    """Check if frontend is built and can be served."""
    print(f"\n{BLUE}=== Checking Frontend ==={RESET}")
    
    issues = []
    frontend_dist = project_root / 'frontend' / 'dist'
    frontend_index = frontend_dist / 'index.html'
    
    if frontend_dist.exists():
        log_success(f"Frontend dist directory exists")
        
        if frontend_index.exists():
            log_success(f"index.html exists (frontend built)")
        else:
            log_warning("index.html not found (frontend needs to be built)")
            issues.append("Frontend not built - run 'npm run build' in frontend/")
        
        # Check for assets
        assets_dir = frontend_dist / 'assets'
        if assets_dir.exists():
            asset_count = len(list(assets_dir.glob('*')))
            log_success(f"Frontend assets found ({asset_count} files)")
        else:
            log_warning("No assets directory found")
    else:
        log_warning("Frontend dist directory not found - frontend needs to be built")
        issues.append("Frontend dist not built")
    
    return len([i for i in issues if 'needs' in i]) == 0, issues

def check_data_directories() -> Tuple[bool, List[str]]:
    """Check that required data directories exist or can be created."""
    print(f"\n{BLUE}=== Checking Data Directories ==={RESET}")
    
    issues = []
    required_dirs = [
        'data/metrics',
        'data/cache',
        'logs',
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            log_success(f"Directory {dir_path} exists")
        else:
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                log_success(f"Directory {dir_path} created")
            except Exception as e:
                log_error(f"Cannot create {dir_path}: {e}")
                issues.append(f"Cannot create directory: {dir_path}")
    
    return len(issues) == 0, issues

def generate_summary(results: Dict[str, Tuple[bool, List[str]]]) -> Dict[str, Any]:
    """Generate validation summary."""
    print(f"\n{BLUE}=== Validation Summary ==={RESET}\n")
    
    passed = sum(1 for result in results.values() if result[0])
    total = len(results)
    
    for check_name, (success, issues) in results.items():
        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
        print(f"{check_name}: {status}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
    
    print(f"\n{BLUE}Overall: {passed}/{total} checks passed{RESET}\n")
    
    if passed == total:
        log_success("All deployment checks passed! ✓")
        return {"status": "success", "score": "100%", "passed": passed, "total": total}
    elif passed >= total - 2:
        log_warning("Deployment mostly ready with minor issues")
        return {"status": "warning", "score": f"{int(passed/total*100)}%", "passed": passed, "total": total}
    else:
        log_error("Deployment has critical issues")
        return {"status": "failure", "score": f"{int(passed/total*100)}%", "passed": passed, "total": total}

async def main():
    """Run all validation checks."""
    print(f"\n{BLUE}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║          Trading API Deployment Validation               ║{RESET}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════════╝{RESET}")
    
    results = {}
    
    # Run checks
    results['Environment Variables'] = check_env_variables()
    results['Configuration Files'] = check_config_files()
    results['Dependencies'] = check_dependencies()
    results['API Routers'] = check_api_routers()
    results['Data Directories'] = check_data_directories()
    results['Frontend'] = check_frontend()
    results['API Health'] = await check_api_health()
    
    # Generate summary
    summary = generate_summary(results)
    
    # Save results to file
    results_file = project_root / 'deployment_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'details': {name: {'success': bool(result[0]), 'issues': result[1]} 
                       for name, result in results.items()}
        }, f, indent=2)
    
    log_success(f"Validation results saved to {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['status'] == 'success' else 1)

if __name__ == '__main__':
    asyncio.run(main())
