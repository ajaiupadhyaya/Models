#!/usr/bin/env python3
"""
Simple launcher for Quantitative Financial Platform
One command to start everything.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the unified platform."""
    print("\n" + "="*80)
    print("🚀 QUANTITATIVE FINANCIAL PLATFORM")
    print("="*80)
    print("\nStarting unified dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("\nPress Ctrl+C to stop\n")
    print("="*80 + "\n")
    
    # Import and run
    from quant_platform import app
    app.run_server(debug=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
