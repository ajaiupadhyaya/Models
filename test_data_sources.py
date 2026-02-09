#!/usr/bin/env python3
"""
Comprehensive test of data fetching from yfinance and verification
that the data pipeline is working correctly.
"""

import os
import sys
from datetime import datetime, timedelta

def test_data_sources():
    """Test all data source functionality."""
    
    print("=" * 60)
    print("DATA SOURCE VERIFICATION TEST")
    print("=" * 60)
    
    from core.data_fetcher import DataFetcher
    
    df = DataFetcher()
    errors = []
    
    # Test 1: Single stock historical data
    print("\n[1/6] Testing single stock data (AAPL)...")
    try:
        data = df.get_stock_data('AAPL', period='5d')
        if data.empty:
            errors.append("Stock data returned empty DataFrame")
            print("   ❌ Empty data returned")
        else:
            print(f"   ✅ Retrieved {len(data)} rows")
            print(f"   ✅ Columns: {list(data.columns)}")
            print(f"   ✅ Date range: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        errors.append(f"Single stock test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Test 2: Multiple stocks
    print("\n[2/6] Testing multiple stocks (AAPL, MSFT, GOOGL)...")
    try:
        multi = df.get_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'], period='5d')
        if multi.empty:
            errors.append("Multiple stocks returned empty DataFrame")
            print("   ❌ Empty data returned")
        else:
            print(f"   ✅ Retrieved data with shape {multi.shape}")
            print(f"   ✅ Tickers: {multi.columns.get_level_values(1).unique().tolist() if multi.columns.nlevels > 1 else 'Single level'}")
    except Exception as e:
        errors.append(f"Multiple stocks test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Test 3: Stock info
    print("\n[3/6] Testing stock info (AAPL)...")
    try:
        info = df.get_stock_info('AAPL')
        if not info:
            errors.append("Stock info returned empty dict")
            print("   ❌ Empty info returned")
        else:
            print(f"   ✅ Company: {info.get('name', 'N/A')}")
            print(f"   ✅ Sector: {info.get('sector', 'N/A')}")
            print(f"   ✅ Market Cap: ${info.get('market_cap', 0) / 1e9:.1f}B")
            print(f"   ✅ Current Price: ${info.get('current_price', 0):.2f}")
    except Exception as e:
        errors.append(f"Stock info test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Test 4: Historical data with date range
    print("\n[4/6] Testing historical data with date range...")
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        data = df.get_stock_data('MSFT', start_date=start_date, end_date=end_date)
        if data.empty:
            errors.append("Historical data with date range returned empty")
            print("   ❌ Empty data returned")
        else:
            print(f"   ✅ Retrieved {len(data)} rows for date range")
            print(f"   ✅ MSFT closing prices: ${data['Close'].iloc[0]:.2f} to ${data['Close'].iloc[-1]:.2f}")
    except Exception as e:
        errors.append(f"Historical data test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Test 5: Crypto data
    print("\n[5/6] Testing crypto data (BTC-USD)...")
    try:
        crypto = df.get_crypto_data('BTC-USD', period='5d')
        if crypto.empty:
            errors.append("Crypto data returned empty")
            print("   ❌ Empty data returned")
        else:
            print(f"   ✅ Retrieved {len(crypto)} rows")
            print(f"   ✅ BTC price: ${crypto['Close'].iloc[-1]:.2f}")
    except Exception as e:
        errors.append(f"Crypto data test failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # Test 6: Economic data (FRED)
    print("\n[6/6] Testing economic data (FRED)...")
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key or not fred_key.strip():
        print("   ⚠️  FRED_API_KEY not configured (optional for economic data)")
        print("   ℹ️  Set FRED_API_KEY in .env file to enable economic indicators")
    else:
        try:
            data = df.get_unemployment_rate()
            if data.empty:
                errors.append("FRED data returned empty")
                print("   ❌ Empty FRED data returned")
            else:
                print(f"   ✅ Unemployment data: {len(data)} data points")
                print(f"   ✅ Latest unemployment rate: {data.iloc[-1]:.1f}%")
        except Exception as e:
            # FRED errors are non-critical
            print(f"   ⚠️  FRED error (non-critical): {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ {len(errors)} CRITICAL ERROR(S) FOUND:")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        print("\n🔴 DATA FETCHING HAS ISSUES - NEEDS FIXING")
        return False
    else:
        print("\n✅ ALL CRITICAL TESTS PASSED!")
        print("✅ yfinance is working correctly")
        print("✅ Stock data, crypto data, and company info are functional")
        print("\n🟢 DATA FETCHING IS OPERATIONAL")
        return True


if __name__ == "__main__":
    success = test_data_sources()
    sys.exit(0 if success else 1)
