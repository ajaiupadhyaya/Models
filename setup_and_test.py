#!/usr/bin/env python3
"""
Comprehensive Setup and Testing Script
Installs dependencies, tests components, fixes issues, and ensures production readiness
"""

import sys
import subprocess
import importlib
from pathlib import Path
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent


def run_command(cmd: list, check: bool = True) -> tuple[bool, str]:
    """Run shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr


def install_dependencies():
    """Install core dependencies."""
    logger.info("Installing core dependencies...")
    
    # Core dependencies first
    core_deps = [
        'numpy>=1.26.0',
        'pandas>=2.1.0',
        'scipy>=1.11.0',
        'scikit-learn>=1.3.0',
    ]
    
    for dep in core_deps:
        logger.info(f"Installing {dep}...")
        success, output = run_command(['pip', 'install', dep], check=False)
        if not success:
            logger.warning(f"Failed to install {dep}: {output[:200]}")
    
    # Financial libraries
    financial_deps = [
        'yfinance>=0.2.28',
        'fredapi>=0.5.1',
        'alpha-vantage>=2.3.1',
    ]
    
    for dep in financial_deps:
        logger.info(f"Installing {dep}...")
        success, output = run_command(['pip', 'install', dep], check=False)
        if not success:
            logger.warning(f"Failed to install {dep}: {output[:200]}")
    
    # Visualization
    viz_deps = [
        'plotly>=5.17.0',
        'dash>=2.14.0',
        'dash-bootstrap-components>=1.5.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
    ]
    
    for dep in viz_deps:
        logger.info(f"Installing {dep}...")
        success, output = run_command(['pip', 'install', dep], check=False)
        if not success:
            logger.warning(f"Failed to install {dep}: {output[:200]}")
    
    # API dependencies
    api_deps = [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'pydantic>=2.5.0',
        'python-multipart>=0.0.6',
        'websockets>=12.0',
    ]
    
    for dep in api_deps:
        logger.info(f"Installing {dep}...")
        success, output = run_command(['pip', 'install', dep], check=False)
        if not success:
            logger.warning(f"Failed to install {dep}: {output[:200]}")
    
    # Utilities
    util_deps = [
        'requests==2.31.0',
        'python-dotenv==1.0.0',
        'tqdm==4.66.1',
        'python-dateutil==2.8.2',
        'pytz==2023.3',
        'schedule>=1.2.0',
    ]
    
    for dep in util_deps:
        logger.info(f"Installing {dep}...")
        success, output = run_command(['pip', 'install', dep], check=False)
        if not success:
            logger.warning(f"Failed to install {dep}: {output[:200]}")
    
    logger.info("Core dependencies installation complete")


def test_imports():
    """Test critical imports."""
    logger.info("Testing critical imports...")
    
    tests = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('scipy', None),
        ('sklearn', None),
        ('yfinance', 'yf'),
        ('plotly', None),
        ('dash', None),
        ('fastapi', None),
    ]
    
    results = {}
    for module_name, alias in tests:
        try:
            mod = importlib.import_module(module_name)
            results[module_name] = True
            logger.info(f"✓ {module_name}")
        except ImportError as e:
            results[module_name] = False
            logger.error(f"✗ {module_name}: {e}")
    
    return results


def test_core_modules():
    """Test core module imports."""
    logger.info("Testing core modules...")
    
    sys.path.insert(0, str(project_root))
    
    tests = [
        ('core.data_fetcher', 'DataFetcher'),
        ('core.utils', None),
        ('core.data_cache', 'DataCache'),
    ]
    
    results = {}
    for module_path, class_name in tests:
        try:
            mod = importlib.import_module(module_path)
            if class_name:
                cls = getattr(mod, class_name)
                # Try instantiation if no required args
                try:
                    instance = cls()
                except:
                    pass  # Some classes need args
            results[module_path] = True
            logger.info(f"✓ {module_path}")
        except Exception as e:
            results[module_path] = False
            logger.error(f"✗ {module_path}: {e}")
            logger.debug(traceback.format_exc())
    
    return results


def fix_issues():
    """Fix any identified issues."""
    logger.info("Checking for issues to fix...")
    
    fixes_applied = []
    
    # Check for common issues
    issues_found = []
    
    # Check if __init__.py files exist where needed
    init_files_needed = [
        'core/__init__.py',
        'models/__init__.py',
        'api/__init__.py',
        'automation/__init__.py',
    ]
    
    for init_file in init_files_needed:
        path = project_root / init_file
        if not path.exists():
            path.write_text('"""Module initialization."""\n')
            fixes_applied.append(f"Created {init_file}")
            logger.info(f"✓ Created {init_file}")
    
    return fixes_applied


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE SETUP AND TESTING")
    logger.info("="*80)
    
    # Step 1: Install dependencies
    logger.info("\n[STEP 1] Installing dependencies...")
    install_dependencies()
    
    # Step 2: Fix issues
    logger.info("\n[STEP 2] Fixing issues...")
    fixes = fix_issues()
    if fixes:
        logger.info(f"Applied {len(fixes)} fixes")
    
    # Step 3: Test imports
    logger.info("\n[STEP 3] Testing imports...")
    import_results = test_imports()
    
    # Step 4: Test core modules
    logger.info("\n[STEP 4] Testing core modules...")
    module_results = test_core_modules()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SETUP SUMMARY")
    logger.info("="*80)
    
    import_passed = sum(1 for v in import_results.values() if v)
    import_total = len(import_results)
    
    module_passed = sum(1 for v in module_results.values() if v)
    module_total = len(module_results)
    
    logger.info(f"Imports: {import_passed}/{import_total} passed")
    logger.info(f"Core Modules: {module_passed}/{module_total} passed")
    logger.info(f"Fixes Applied: {len(fixes)}")
    
    if import_passed == import_total and module_passed == module_total:
        logger.info("\n✅ SETUP COMPLETE - ALL TESTS PASSED")
        return 0
    else:
        logger.warning("\n⚠️ SETUP COMPLETE - SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
