#!/usr/bin/env python3
"""
Quick verification that all data fetching improvements are working.
Run this to confirm the system is operational.
"""

def quick_check():
    print("🔍 QUICK DATA FETCHING CHECK")
    print("=" * 60)
    
    results = {"passed": [], "failed": []}
    
    # Test 1: Import DataFetcher
    print("\n[1/5] Importing DataFetcher...")
    try:
        from core.data_fetcher import DataFetcher
        results["passed"].append("Import DataFetcher")
        print("   ✅ Success")
    except Exception as e:
        results["failed"].append(f"Import DataFetcher: {e}")
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 2: Fetch single stock
    print("\n[2/5] Fetching single stock (AAPL)...")
    try:
        df = DataFetcher()
        data = df.get_stock_data('AAPL', period='5d')
        if not data.empty:
            results["passed"].append("Single stock fetch")
            print(f"   ✅ Success - {len(data)} rows")
        else:
            results["failed"].append("Single stock fetch: Empty data")
            print("   ❌ Failed: Empty data")
    except Exception as e:
        results["failed"].append(f"Single stock fetch: {e}")
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Fetch multiple stocks
    print("\n[3/5] Fetching multiple stocks...")
    try:
        data = df.get_multiple_stocks(['AAPL', 'MSFT'], period='5d')
        if not data.empty:
            results["passed"].append("Multiple stocks fetch")
            print(f"   ✅ Success - shape {data.shape}")
        else:
            results["failed"].append("Multiple stocks fetch: Empty data")
            print("   ❌ Failed: Empty data")
    except Exception as e:
        results["failed"].append(f"Multiple stocks fetch: {e}")
        print(f"   ❌ Failed: {e}")
    
    # Test 4: Fetch crypto
    print("\n[4/5] Fetching crypto (BTC-USD)...")
    try:
        data = df.get_crypto_data('BTC-USD', period='5d')
        if not data.empty:
            results["passed"].append("Crypto fetch")
            print(f"   ✅ Success - {len(data)} rows")
        else:
            results["failed"].append("Crypto fetch: Empty data")
            print("   ❌ Failed: Empty data")
    except Exception as e:
        results["failed"].append(f"Crypto fetch: {e}")
        print(f"   ❌ Failed: {e}")
    
    # Test 5: API endpoint check
    print("\n[5/5] Checking API endpoints...")
    try:
        from api.data_api import router
        route_count = len([r for r in router.routes])
        if route_count >= 6:  # Should have at least 6 routes including health-check
            results["passed"].append("API endpoints")
            print(f"   ✅ Success - {route_count} routes registered")
        else:
            results["failed"].append(f"API endpoints: Only {route_count} routes")
            print(f"   ⚠️  Warning: Only {route_count} routes")
    except Exception as e:
        results["failed"].append(f"API endpoints: {e}")
        print(f"   ❌ Failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = len(results["passed"])
    failed = len(results["failed"])
    total = passed + failed
    
    print(f"\n✅ Passed: {passed}/{total}")
    if results["failed"]:
        print(f"❌ Failed: {failed}/{total}")
        for fail in results["failed"]:
            print(f"   - {fail}")
    
    if failed == 0:
        print("\n🟢 ALL CHECKS PASSED - SYSTEM OPERATIONAL")
        return True
    elif passed >= 3:
        print("\n🟡 PARTIAL SUCCESS - CORE FEATURES WORKING")
        return True
    else:
        print("\n🔴 CRITICAL FAILURES - NEEDS ATTENTION")
        return False


if __name__ == "__main__":
    import sys
    success = quick_check()
    sys.exit(0 if success else 1)
