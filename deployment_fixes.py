#!/usr/bin/env python3
"""
Production Deployment Fixes

Addresses common deployment issues:
1. Missing environment variables
2. Frontend build
3. Configuration validation
4. Database initialization
5. Error handling improvements
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def log_success(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")

def log_error(msg: str):
    print(f"{RED}✗ {msg}{RESET}")

def log_warning(msg: str):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def log_info(msg: str):
    print(f"{BLUE}ℹ {msg}{RESET}")

def ensure_env_file():
    """Create .env file if missing with defaults."""
    print(f"\n{BLUE}=== Ensuring Environment Configuration ==={RESET}")
    
    env_file = project_root / '.env'
    env_example = project_root / '.env.example'
    
    if env_file.exists():
        log_success(".env file already exists")
        return
    
    if env_example.exists():
        # Copy example to .env
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            log_success(".env file created from .env.example")
        except Exception as e:
            log_error(f"Failed to create .env: {e}")
            return
    
    # Create minimal .env if not present
    if not env_file.exists():
        minimal_env = """# Trading API Environment Configuration

# Required for login and core app
TERMINAL_USER=demo
TERMINAL_PASSWORD=demo123
AUTH_SECRET=change-me-in-production-use-long-random-string-at-least-32-chars

# Data providers (optional but recommended for full functionality)
# Get free keys at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=

# Get free key at: https://www.alphavantage.co/
ALPHA_VANTAGE_API_KEY=

# AI/LLM (optional for AI tab)
# Get key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Optional: News data
# Get key at: https://finnhub.io/
FINNHUB_API_KEY=

# Optional: Paper Trading
ENABLE_PAPER_TRADING=false
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_API_BASE=https://paper-api.alpaca.markets

# Optional: Feature flags
ENABLE_METRICS=true
WEBSOCKET_ENABLED=true
"""
        try:
            with open(env_file, 'w') as f:
                f.write(minimal_env)
            log_success(".env file created with defaults")
            log_warning("Update TERMINAL_USER, TERMINAL_PASSWORD, and AUTH_SECRET for production")
        except Exception as e:
            log_error(f"Failed to create .env: {e}")

def ensure_directories():
    """Create required data directories."""
    print(f"\n{BLUE}=== Ensuring Required Directories ==={RESET}")
    
    required_dirs = [
        'data',
        'data/metrics',
        'data/cache',
        'data/models',
        'logs',
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            log_success(f"Directory {dir_name}/ ready")
        except Exception as e:
            log_error(f"Cannot create {dir_name}/: {e}")

def build_frontend():
    """Build React TypeScript frontend if needed."""
    print(f"\n{BLUE}=== Checking Frontend Build ==={RESET}")
    
    frontend_dir = project_root / 'frontend'
    dist_dir = frontend_dir / 'dist'
    index_html = dist_dir / 'index.html'
    
    if index_html.exists():
        log_success("Frontend already built (dist/index.html exists)")
        return True
    
    log_warning("Frontend not built, attempting to build...")
    
    if not frontend_dir.exists():
        log_error("Frontend directory not found at frontend/")
        return False
    
    try:
        # Check if node_modules exists
        node_modules = frontend_dir / 'node_modules'
        if not node_modules.exists():
            log_info("Installing npm dependencies...")
            result = subprocess.run(
                ['npm', 'ci'],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                log_error(f"npm ci failed: {result.stderr[:100]}")
                return False
            log_success("npm dependencies installed")
        
        # Build frontend
        log_info("Building frontend with npm run build...")
        result = subprocess.run(
            ['npm', 'run', 'build'],
            cwd=str(frontend_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            log_error(f"npm build failed: {result.stderr[:200]}")
            return False
        
        log_success("Frontend built successfully")
        return True
    
    except subprocess.TimeoutExpired:
        log_error("Frontend build timed out")
        return False
    except Exception as e:
        log_error(f"Frontend build error: {e}")
        return False

def validate_configuration():
    """Validate critical configuration."""
    print(f"\n{BLUE}=== Validating Configuration ==={RESET}")
    
    issues = []
    
    # Check auth secret
    auth_secret = os.getenv('AUTH_SECRET')
    if not auth_secret or len(auth_secret) < 16:
        log_warning("AUTH_SECRET is too short (should be >=16 characters)")
        issues.append("AUTH_SECRET too short")
    elif 'change-me' in auth_secret.lower():
        log_warning("AUTH_SECRET still has default value - should be changed for production")
        issues.append("AUTH_SECRET not customized")
    else:
        log_success("AUTH_SECRET is configured")
    
    # Check terminal credentials
    if os.getenv('TERMINAL_USER') and os.getenv('TERMINAL_PASSWORD'):
        log_success("Terminal credentials configured")
    else:
        log_warning("Terminal credentials not properly configured")
        issues.append("Terminal credentials missing")
    
    # Check optional API keys
    optional_keys = ['FRED_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'OPENAI_API_KEY']
    configured = sum(1 for k in optional_keys if os.getenv(k))
    log_info(f"{configured}/{len(optional_keys)} optional API keys configured")
    
    return len([i for i in issues if 'missing' in i.lower()]) == 0

def generate_requirements_lock():
    """Ensure requirements files are complete."""
    print(f"\n{BLUE}=== Checking Requirements Files ==={RESET}")
    
    required_files = [
        'requirements-api.txt',
        'requirements.txt',
    ]
    
    for req_file in required_files:
        path = project_root / req_file
        if path.exists():
            log_success(f"{req_file} exists")
        else:
            log_warning(f"{req_file} not found")

def create_deployment_success_check():
    """Create a script to check if deployment succeeded."""
    check_script = """#!/bin/bash
# Check if API is running and responding

echo "Checking API health..."

# Try health endpoint
HEALTH=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8000/health 2>/dev/null)

if [ "$HEALTH" = "200" ]; then
    echo "✓ API is healthy (HTTP 200)"
    
    # Check info endpoint
    INFO=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8000/info 2>/dev/null)
    if [ "$INFO" = "200" ]; then
        echo "✓ API info endpoint working"
    else
        echo "⚠ API info endpoint returned HTTP $INFO"
    fi
    
    exit 0
else
    echo "✗ API health check failed (HTTP $HEALTH)"
    exit 1
fi
"""
    
    check_path = project_root / 'check_deployment.sh'
    try:
        with open(check_path, 'w') as f:
            f.write(check_script)
        os.chmod(check_path, 0o755)
        log_success(f"Deployment check script created: {check_path}")
    except Exception as e:
        log_warning(f"Could not create check script: {e}")

def create_production_checklist():
    """Create a production deployment checklist."""
    checklist = {
        "pre_deployment": [
            "✓ All environment variables configured (AUTH_SECRET, TERMINAL_USER, TERMINAL_PASSWORD)",
            "✓ API keys added for required data providers (FRED_API_KEY, ALPHA_VANTAGE_API_KEY)",
            "✓ Frontend built and assets bundled",
            "✓ All tests passing locally",
            "✓ Docker image builds without errors",
            "✓ CORS configuration verified",
            "✓ Database/cache connections working",
        ],
        "deployment": [
            "Trigger deployment in Render dashboard",
            "Monitor build logs for errors",
            "Wait for health check (usually 2-5 minutes)",
            "Verify /health endpoint returns 200 OK",
            "Verify /info endpoint shows routers_loaded",
            "Test login with configured credentials",
            "Check API logs for any startup errors",
        ],
        "post_deployment": [
            "Verify all API endpoints respond (queries, risk, backtesting)",
            "Test WebSocket connection (any ticker subscription)",
            "Verify frontend loads and is responsive",
            "Check that data is loading (quotes, charts)",
            "Monitor logs for errors or warnings",
            "Set up error alerting (Render Notifications)",
        ],
        "critical_endpoints": [
            "GET /health - Returns 200 (load balancer)",
            "GET /info - Shows routers_loaded and capabilities",
            "GET /api/v1/data/quotes - Needs yfinance working",
            "GET /api/v1/monitoring/dashboard - System metrics",
            "WebSocket ws://*/api/v1/ws/prices/{symbol} - Live prices",
            "POST /api/v1/predictions/predict - Model predictions",
        ],
    }
    
    checklist_file = project_root / 'PRODUCTION_CHECKLIST.md'
    try:
        with open(checklist_file, 'w') as f:
            f.write("# Production Deployment Checklist\n\n")
            
            for section, items in checklist.items():
                section_title = section.replace('_', ' ').title()
                f.write(f"## {section_title}\n\n")
                for item in items:
                    f.write(f"- [ ] {item}\n")
                f.write("\n")
        
        log_success(f"Production checklist created: {checklist_file}")
    except Exception as e:
        log_warning(f"Could not create checklist: {e}")

def main():
    """Run all deployment fixes."""
    print(f"\n{BLUE}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║        Production Deployment Preparation                 ║{RESET}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════════╝{RESET}")
    
    # Run checks
    ensure_env_file()
    ensure_directories()
    frontend_ok = build_frontend()
    config_ok = validate_configuration()
    generate_requirements_lock()
    create_deployment_success_check()
    create_production_checklist()
    
    print(f"\n{BLUE}=== Deployment Preparation Complete ==={RESET}\n")
    
    if frontend_ok and config_ok:
        log_success("All critical preparations complete!")
        log_info("Next steps:")
        print("  1. Push changes to GitHub: git push origin main")
        print("  2. Trigger deployment in Render: 'Deploy latest commit'")
        print("  3. Monitor logs for startup errors")
        print("  4. Verify /health endpoint returns 200")
        print("  5. Check deployment_validation.py for full status")
    else:
        log_warning("Some preparations need attention (see above)")
    
    print()

if __name__ == '__main__':
    main()
