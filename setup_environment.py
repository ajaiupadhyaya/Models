#!/usr/bin/env python3
"""
Comprehensive Environment Setup Script
Sets up and validates the entire environment for production use
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Comprehensive environment setup and validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = []
    
    def print_step(self, step: str):
        """Print step header."""
        print(f"\n{BOLD}{CYAN}{'='*80}")
        print(f"{step}")
        print(f"{'='*80}{END}\n")
    
    def check_python_version(self) -> bool:
        """Check Python version."""
        self.print_step("Checking Python Version")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"{GREEN}✓{END} Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print(f"{RED}✗{END} Python {version.major}.{version.minor} (requires 3.8+)")
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        self.print_step("Creating Directory Structure")
        
        directories = [
            "data/pipeline",
            "data/models",
            "data/cache",
            "data/metrics",
            "logs",
            "automation",
            "outputs",
            "reports",
            "presentations"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"{GREEN}✓{END} {dir_path}")
    
    def install_dependencies(self, upgrade_pip: bool = True) -> bool:
        """Install dependencies."""
        self.print_step("Installing Dependencies")
        
        try:
            if upgrade_pip:
                print("Upgrading pip...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                                    stdout=subprocess.DEVNULL)
            
            print("Installing requirements...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                                 str(self.project_root / "requirements.txt")],
                                stdout=subprocess.DEVNULL)
            
            print(f"{GREEN}✓{END} Dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{RED}✗{END} Failed to install dependencies: {e}")
            return False
    
    def create_env_file(self):
        """Create .env file if it doesn't exist."""
        self.print_step("Setting Up Environment Configuration")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists():
            print(f"{GREEN}✓{END} .env file already exists")
            return
        
        # Create from example if available
        if env_example.exists():
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print(f"{GREEN}✓{END} Created .env from template")
        else:
            # Create default .env
            default_env = """# API Keys (Optional - many features work without them)
# Get FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=

# Get Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=

# Trading API Keys (Optional - for paper trading)
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Configuration
ENABLE_PAPER_TRADING=false
"""
            with open(env_file, 'w') as f:
                f.write(default_env)
            print(f"{GREEN}✓{END} Created default .env file")
        
        print(f"{YELLOW}⚠{END} Please edit .env and add your API keys (optional)")
    
    def create_automation_config(self):
        """Create automation configuration."""
        self.print_step("Setting Up Automation Configuration")
        
        config_file = self.project_root / "automation" / "config.json"
        
        if config_file.exists():
            print(f"{GREEN}✓{END} Automation config already exists")
            return
        
        default_config = {
            "max_workers": 4,
            "data_update_frequency": "daily",
            "ml_retrain_frequency": "weekly",
            "trading_enabled": False,
            "monitoring_enabled": True,
            "alert_thresholds": {
                "data_quality": 0.95,
                "model_accuracy": 0.70,
                "portfolio_drawdown": 0.10
            },
            "symbols": ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "GOOGL"],
            "economic_indicators": ["UNRATE", "GDP", "CPIAUCSL", "FEDFUNDS", "DGS10", "PAYEMS"]
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"{GREEN}✓{END} Created automation config")
    
    def test_imports(self) -> bool:
        """Test critical imports."""
        self.print_step("Testing Imports")
        
        imports = [
            ("pandas", "Data processing"),
            ("numpy", "Numerical computing"),
            ("plotly", "Visualizations"),
            ("yfinance", "Market data"),
            ("fastapi", "API framework"),
            ("dash", "Dashboard"),
            ("sklearn", "Machine learning"),
        ]
        
        all_passed = True
        for module, desc in imports:
            try:
                __import__(module)
                print(f"{GREEN}✓{END} {module:20} - {desc}")
            except ImportError:
                print(f"{RED}✗{END} {module:20} - NOT INSTALLED")
                all_passed = False
        
        return all_passed
    
    def test_core_functionality(self) -> bool:
        """Test core functionality."""
        self.print_step("Testing Core Functionality")
        
        tests = []
        
        # Test data fetcher
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            data = fetcher.get_stock_data('AAPL', period='1mo')
            if len(data) > 0:
                print(f"{GREEN}✓{END} Data fetching works")
                tests.append(True)
            else:
                print(f"{RED}✗{END} Data fetching returned no data")
                tests.append(False)
        except Exception as e:
            print(f"{RED}✗{END} Data fetching failed: {e}")
            tests.append(False)
        
        # Test DCF model
        try:
            from models.valuation.dcf_model import DCFModel
            dcf = DCFModel([100, 120, 140], terminal_growth_rate=0.03, wacc=0.10)
            ev = dcf.calculate_enterprise_value()
            print(f"{GREEN}✓{END} DCF model works (EV=${ev:,.2f})")
            tests.append(True)
        except Exception as e:
            print(f"{RED}✗{END} DCF model failed: {e}")
            tests.append(False)
        
        # Test Black-Scholes
        try:
            from models.options.black_scholes import BlackScholes
            price = BlackScholes.call_price(100, 100, 0.25, 0.05, 0.20)
            print(f"{GREEN}✓{END} Black-Scholes works (Call=${price:.2f})")
            tests.append(True)
        except Exception as e:
            print(f"{RED}✗{END} Black-Scholes failed: {e}")
            tests.append(False)
        
        return all(tests)
    
    def run_setup(self, install_deps: bool = True):
        """Run complete setup."""
        print(f"\n{BOLD}{BLUE}{'='*80}")
        print("COMPREHENSIVE ENVIRONMENT SETUP")
        print(f"{'='*80}{END}\n")
        
        # Check Python
        if not self.check_python_version():
            print(f"\n{RED}Python version check failed. Please upgrade Python.{END}\n")
            return False
        
        # Create directories
        self.create_directories()
        
        # Create config files
        self.create_env_file()
        self.create_automation_config()
        
        # Install dependencies
        if install_deps:
            if not self.install_dependencies():
                print(f"\n{YELLOW}Dependency installation had issues. Continuing...{END}\n")
        
        # Test imports
        imports_ok = self.test_imports()
        
        # Test functionality
        functionality_ok = self.test_core_functionality()
        
        # Summary
        print(f"\n{BOLD}{'='*80}")
        print("SETUP SUMMARY")
        print(f"{'='*80}{END}\n")
        
        print(f"Python Version: {GREEN}✓{END}")
        print(f"Directories: {GREEN}✓{END}")
        print(f"Configuration: {GREEN}✓{END}")
        print(f"Imports: {'✓' if imports_ok else '✗'}")
        print(f"Functionality: {'✓' if functionality_ok else '✗'}")
        
        if imports_ok and functionality_ok:
            print(f"\n{GREEN}{BOLD}✓ Environment setup complete!{END}\n")
            print(f"{CYAN}Next steps:{END}")
            print(f"  1. Edit .env file and add API keys (optional)")
            print(f"  2. Run: {BOLD}python launch.py{END}")
            print(f"  3. Or start automation: {BOLD}python -m automation.orchestrator{END}\n")
            return True
        else:
            print(f"\n{YELLOW}⚠ Setup complete with some issues{END}\n")
            print(f"{CYAN}You may need to:{END}")
            print(f"  - Install missing dependencies: {BOLD}pip install -r requirements.txt{END}")
            print(f"  - Check error messages above\n")
            return False


def main():
    """Main setup function."""
    project_root = Path(__file__).parent
    
    setup = EnvironmentSetup(project_root)
    
    # Ask about dependencies
    install_deps = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-install':
        install_deps = False
    
    success = setup.run_setup(install_deps=install_deps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
