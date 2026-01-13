#!/usr/bin/env python3
"""
Environment Validation Script
Comprehensive verification of project environment and setup
"""

import sys
import subprocess
from pathlib import Path

class EnvironmentValidator:
    """Validate project environment."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, condition: bool, message: str = ""):
        """Record test result."""
        status = "✓ PASS" if condition else "✗ FAIL"
        self.results.append(f"{status:8} | {name:40} | {message}")
        
        if condition:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_results(self):
        """Print all results."""
        print("\n" + "="*100)
        print("ENVIRONMENT VALIDATION REPORT")
        print("="*100 + "\n")
        
        for result in self.results:
            print(result)
        
        print("\n" + "="*100)
        print(f"Results: {self.passed} Passed, {self.failed} Failed")
        print("="*100 + "\n")
        
        return self.failed == 0
    
    def run_all(self):
        """Run all validation tests."""
        print("Running environment validation...\n")
        
        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.test("Python version", sys.version_info >= (3, 10), py_version)
        
        # Check project structure
        project_root = Path("/Users/ajaiupadhyaya/Documents/Models")
        self.test("Project root exists", project_root.exists(), str(project_root))
        
        # Check critical directories
        dirs = {
            "models": project_root / "models",
            "core": project_root / "core",
            "api": project_root / "api",
            "notebooks": project_root / "notebooks",
            "data": project_root / "data"
        }
        
        for name, path in dirs.items():
            self.test(f"Directory: {name}", path.exists(), str(path))
        
        # Check critical files
        files = {
            "core/backtesting.py": project_root / "core" / "backtesting.py",
            "models/ml/advanced_trading.py": project_root / "models" / "ml" / "advanced_trading.py",
            "api/main.py": project_root / "api" / "main.py",
            "requirements.txt": project_root / "requirements.txt",
        }
        
        for name, path in files.items():
            self.test(f"File: {name}", path.exists(), str(path))
        
        # Python imports - Core
        core_imports = [
            ("core.backtesting.BacktestEngine", "backtesting engine"),
            ("core.backtesting.SimpleMLPredictor", "simple ML predictor"),
            ("core.backtesting.WalkForwardAnalysis", "walk-forward validation"),
        ]
        
        for import_path, desc in core_imports:
            try:
                parts = import_path.split(".")
                module_name = ".".join(parts[:-1])
                class_name = parts[-1]
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.test(f"Import: {import_path}", True, desc)
            except Exception as e:
                self.test(f"Import: {import_path}", False, str(e)[:50])
        
        # Python imports - ML Models
        ml_imports = [
            ("models.ml.advanced_trading.LSTMPredictor", "LSTM deep learning"),
            ("models.ml.advanced_trading.EnsemblePredictor", "ensemble ML"),
            ("models.ml.advanced_trading.RLReadyEnvironment", "RL environment"),
        ]
        
        for import_path, desc in ml_imports:
            try:
                parts = import_path.split(".")
                module_name = ".".join(parts[:-1])
                class_name = parts[-1]
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.test(f"Import: {import_path}", True, desc)
            except Exception as e:
                self.test(f"Import: {import_path}", False, str(e)[:50])
        
        # Python packages
        packages = [
            ("fastapi", "FastAPI web framework"),
            ("uvicorn", "ASGI server"),
            ("pydantic", "data validation"),
            ("websockets", "WebSocket support"),
            ("yfinance", "market data"),
            ("pandas", "data processing"),
            ("numpy", "numerical computing"),
            ("sklearn", "scikit-learn ML"),
            ("tensorflow", "deep learning"),
            ("keras", "neural networks"),
        ]
        
        for pkg, desc in packages:
            try:
                __import__(pkg)
                self.test(f"Package: {pkg}", True, desc)
            except ImportError:
                self.test(f"Package: {pkg}", False, f"not installed")
        
        # API server check
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            self.test("API server running", response.status_code == 200, "http://localhost:8000")
        except Exception as e:
            self.test("API server running", False, str(e)[:50])
        
        # Permissions
        self.test("Write access to data/", (project_root / "data").is_dir() and 
                 (project_root / "data").stat().st_mode & 0o200, "read/write")
        
        # Notebook files
        notebooks = [
            "01_getting_started.ipynb",
            "10_ml_backtesting.ipynb",
            "11_rl_trading_agents.ipynb",
            "12_lstm_deep_learning.ipynb",
            "13_multi_asset_strategies.ipynb",
        ]
        
        notebooks_path = project_root / "notebooks"
        for nb in notebooks:
            nb_path = notebooks_path / nb
            self.test(f"Notebook: {nb}", nb_path.exists(), str(nb_path))
        
        # API modules
        api_modules = [
            "api/__init__.py",
            "api/main.py",
            "api/models_api.py",
            "api/predictions_api.py",
            "api/backtesting_api.py",
            "api/websocket_api.py",
            "api/monitoring.py",
        ]
        
        for module in api_modules:
            module_path = project_root / module
            self.test(f"API module: {module}", module_path.exists(), str(module_path))
        
        return self.print_results()


if __name__ == "__main__":
    validator = EnvironmentValidator()
    success = validator.run_all()
    
    sys.exit(0 if success else 1)
