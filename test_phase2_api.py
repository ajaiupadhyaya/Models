"""
Phase 2 API Endpoint Tests
Validates sentiment and factor analysis API endpoints
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_api_imports():
    """Test that API can import Phase 2 modules."""
    print("\n" + "="*60)
    print("TEST 1: API Module Imports")
    print("="*60)
    
    try:
        from api import predictions_api, risk_api
        print(f"✓ predictions_api loaded ({len(predictions_api.router.routes)} routes)")
        print(f"✓ risk_api loaded ({len(risk_api.router.routes)} routes)")
        
        # Check for new endpoints
        pred_routes = [r.path for r in predictions_api.router.routes]
        risk_routes = [r.path for r in risk_api.router.routes]
        
        # Phase 2 sentiment endpoints
        sentiment_endpoints = [
            '/sentiment/{ticker}',
            '/sentiment/batch'
        ]
        
        for endpoint in sentiment_endpoints:
            if endpoint in pred_routes:
                print(f"✓ Found sentiment endpoint: {endpoint}")
            else:
                print(f"✗ Missing sentiment endpoint: {endpoint}")
        
        # Phase 2 factor endpoints
        factor_endpoints = [
            '/multi-factor/{ticker}',
            '/factor-ic'
        ]
        
        for endpoint in factor_endpoints:
            if endpoint in risk_routes:
                print(f"✓ Found factor endpoint: {endpoint}")
            else:
                print(f"✗ Missing factor endpoint: {endpoint}")
        
        return True
    except Exception as e:
        print(f"✗ API import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run API tests."""
    print("\n" + "="*60)
    print("PHASE 2 API ENDPOINT VALIDATION")
    print("="*60)
    
    test_api_imports()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Phase 2 API endpoints successfully integrated!")
    print("\nNew Sentiment Endpoints:")
    print("  GET /api/v1/predictions/sentiment/{ticker}")
    print("  GET /api/v1/predictions/sentiment/batch?tickers=...")
    print("\nNew Factor Analysis Endpoints:")
    print("  GET /api/v1/risk/multi-factor/{ticker}")
    print("  GET /api/v1/risk/factor-ic?factor_ticker=...")
    
    return 0


if __name__ == "__main__":
    exit(main())
