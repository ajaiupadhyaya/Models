"""
Run the interactive financial dashboard.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.dashboard import create_dashboard

if __name__ == '__main__':
    print("=" * 60)
    print("Financial Analysis Dashboard")
    print("=" * 60)
    print("\nStarting dashboard server...")
    print("Open your browser to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    dashboard = create_dashboard()
    dashboard.run(debug=True, port=8050)
