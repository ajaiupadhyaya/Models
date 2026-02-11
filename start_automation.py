#!/usr/bin/env python3
"""
Master Automation Startup Script
Starts all automated processes with real data
"""

import sys
import os
import signal
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automation.orchestrator import AutomationOrchestrator
from automation.data_pipeline import DataPipeline
from automation.ml_pipeline import MLPipeline

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"automation_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global orchestrator
orchestrator = None


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print(f"\n\n{YELLOW}Shutting down automation...{END}")
    if orchestrator:
        orchestrator.stop_all()
    print(f"{GREEN}Shutdown complete.{END}\n")
    sys.exit(0)


def print_header():
    """Print startup header."""
    print(f"\n{BOLD}{CYAN}{'='*80}")
    print("FINANCIAL MODELS - AUTOMATED SYSTEM")
    print(f"{'='*80}{END}\n")
    print(f"{BLUE}Starting all automated processes...{END}\n")


def initialize_data_pipeline():
    """Initialize and test data pipeline with real data."""
    print(f"{CYAN}Initializing Data Pipeline...{END}")
    
    try:
        pipeline = DataPipeline()
        
        # Test with real data
        print(f"  Testing data fetching with real symbols...")
        test_symbols = ["SPY", "AAPL"]
        results = pipeline.batch_fetch_stocks(test_symbols, period="1mo")
        
        successful = results['summary']['successful']
        total = results['summary']['total']
        
        if successful == total:
            print(f"  {GREEN}✓{END} Data pipeline working ({successful}/{total} successful)")
        else:
            print(f"  {YELLOW}⚠{END} Data pipeline partial ({successful}/{total} successful)")
        
        return pipeline
        
    except Exception as e:
        print(f"  {RED}✗{END} Data pipeline error: {e}")
        logger.error(f"Data pipeline initialization error: {e}")
        return None


def initialize_ml_pipeline():
    """Initialize ML pipeline."""
    print(f"{CYAN}Initializing ML Pipeline...{END}")
    
    try:
        ml_pipeline = MLPipeline()
        print(f"  {GREEN}✓{END} ML pipeline ready")
        return ml_pipeline
        
    except Exception as e:
        print(f"  {RED}✗{END} ML pipeline error: {e}")
        logger.error(f"ML pipeline initialization error: {e}")
        return None


def main():
    """Main automation startup."""
    global orchestrator
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_header()
    
    # Initialize orchestrator
    print(f"{CYAN}Initializing Automation Orchestrator...{END}")
    try:
        orchestrator = AutomationOrchestrator()
        print(f"  {GREEN}✓{END} Orchestrator initialized\n")
    except Exception as e:
        print(f"  {RED}✗{END} Orchestrator initialization failed: {e}")
        logger.error(f"Orchestrator error: {e}")
        return 1
    
    # Initialize components
    data_pipeline = initialize_data_pipeline()
    ml_pipeline = initialize_ml_pipeline()
    
    print()
    
    # Start all processes
    print(f"{BOLD}{GREEN}Starting all automation processes...{END}\n")
    orchestrator.start_all()
    
    # Print status
    status = orchestrator.get_status()
    print(f"{CYAN}Automation Status:{END}")
    print(f"  Running: {GREEN if status['is_running'] else RED}{status['is_running']}{END}")
    print(f"  Processes: {len(status['processes'])}")
    print(f"  Scheduled Jobs: {status['scheduler_status']['total_jobs']}\n")
    
    print(f"{BOLD}{GREEN}✓ All systems operational!{END}\n")
    print(f"{CYAN}Automation is running. Press Ctrl+C to stop.{END}\n")
    print(f"{BLUE}Logs: {log_file}{END}\n")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(60)
            
            # Print periodic status
            status = orchestrator.get_status()
            if status['is_running']:
                print(f"{CYAN}[{datetime.now().strftime('%H:%M:%S')}] System running - {len(status['processes'])} processes active{END}")
            
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
