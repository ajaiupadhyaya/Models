#!/usr/bin/env python3
"""
Production Launch Script
Launches all services and ensures automation is running
"""

import sys
import subprocess
import time
import logging
from pathlib import Path
import signal
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ProductionLauncher:
    """Launch production services."""
    
    def __init__(self):
        """Initialize launcher."""
        self.processes = []
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received, stopping services...")
        self.running = False
        self.stop_all()
        sys.exit(0)
    
    def start_api_server(self, port: int = 8000):
        """Start API server."""
        logger.info(f"Starting API server on port {port}...")
        try:
            process = subprocess.Popen(
                [sys.executable, 'api/main.py'],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(('API Server', process))
            logger.info(f"✓ API server started (PID: {process.pid})")
            logger.info(f"  API available at: http://localhost:{port}")
            logger.info(f"  API docs at: http://localhost:{port}/docs")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to start API server: {e}")
            return False
    
    def start_dashboard(self, port: int = 8050):
        """Start dashboard."""
        logger.info(f"Starting dashboard on port {port}...")
        try:
            process = subprocess.Popen(
                [sys.executable, 'start.py'],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(('Dashboard', process))
            logger.info(f"✓ Dashboard started (PID: {process.pid})")
            logger.info(f"  Dashboard available at: http://localhost:{port}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to start dashboard: {e}")
            return False
    
    def start_automation(self):
        """Start automation orchestrator."""
        logger.info("Starting automation orchestrator...")
        try:
            from automation.orchestrator import AutomationOrchestrator
            
            orchestrator = AutomationOrchestrator()
            orchestrator.start_all()
            
            logger.info("✓ Automation orchestrator started")
            logger.info("  - Data pipeline: Running")
            logger.info("  - ML training pipeline: Running")
            logger.info("  - Monitoring: Running")
            
            return orchestrator
        except Exception as e:
            logger.error(f"✗ Failed to start automation: {e}")
            return None
    
    def check_health(self):
        """Check health of all services."""
        logger.info("Checking service health...")
        
        # Check API
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=2)
            if response.status_code == 200:
                logger.info("✓ API server is healthy")
            else:
                logger.warning(f"⚠ API server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠ Could not check API health: {e}")
        
        # Check processes
        for name, process in self.processes:
            if process.poll() is None:
                logger.info(f"✓ {name} is running (PID: {process.pid})")
            else:
                logger.error(f"✗ {name} has stopped")
    
    def stop_all(self):
        """Stop all services."""
        logger.info("Stopping all services...")
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✓ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"Force killed {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
    
    def launch_all(self, start_api: bool = True, start_dashboard: bool = True, start_automation: bool = True):
        """Launch all services."""
        logger.info("="*80)
        logger.info("PRODUCTION LAUNCH")
        logger.info("="*80)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        success_count = 0
        
        # Start API
        if start_api:
            if self.start_api_server():
                success_count += 1
            time.sleep(2)  # Give API time to start
        
        # Start Dashboard
        if start_dashboard:
            if self.start_dashboard():
                success_count += 1
            time.sleep(2)
        
        # Start Automation
        orchestrator = None
        if start_automation:
            orchestrator = self.start_automation()
            if orchestrator:
                success_count += 1
        
        # Health check
        time.sleep(3)
        self.check_health()
        
        logger.info("\n" + "="*80)
        logger.info("PRODUCTION SERVICES RUNNING")
        logger.info("="*80)
        logger.info(f"Services started: {success_count}/3")
        logger.info("\nPress Ctrl+C to stop all services")
        logger.info("="*80 + "\n")
        
        # Keep running
        try:
            while self.running:
                time.sleep(1)
                # Check if processes are still running
                for name, process in self.processes[:]:
                    if process.poll() is not None:
                        logger.warning(f"{name} has stopped unexpectedly")
                        self.processes.remove((name, process))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()
            if orchestrator:
                try:
                    orchestrator.stop_all()
                except:
                    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch production services')
    parser.add_argument('--no-api', action='store_true', help='Skip API server')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip dashboard')
    parser.add_argument('--no-automation', action='store_true', help='Skip automation')
    
    args = parser.parse_args()
    
    launcher = ProductionLauncher()
    launcher.launch_all(
        start_api=not args.no_api,
        start_dashboard=not args.no_dashboard,
        start_automation=not args.no_automation
    )
