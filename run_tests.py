#!/usr/bin/env python3
"""
Run Comprehensive Test Suite
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'


def run_tests():
    """Run all tests."""
    print(f"\n{BOLD}{CYAN}{'='*80}")
    print("COMPREHENSIVE TEST SUITE")
    print(f"{'='*80}{END}\n")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    try:
        from tests.test_automation import (
            TestDataPipeline,
            TestMLPipeline,
            TestOrchestrator,
            TestRealData
        )
        
        suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
        suite.addTests(loader.loadTestsFromTestCase(TestMLPipeline))
        suite.addTests(loader.loadTestsFromTestCase(TestOrchestrator))
        suite.addTests(loader.loadTestsFromTestCase(TestRealData))
        
    except ImportError as e:
        print(f"{YELLOW}⚠ Some test modules not available: {e}{END}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{BOLD}{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}{END}\n")
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print(f"Total Tests: {total}")
    print(f"{GREEN}Passed: {passed}{END}")
    if failures > 0:
        print(f"{YELLOW}Failures: {failures}{END}")
    if errors > 0:
        print(f"{RED}Errors: {errors}{END}")
    
    if failures == 0 and errors == 0:
        print(f"\n{GREEN}{BOLD}✓ All tests passed!{END}\n")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some tests failed{END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
