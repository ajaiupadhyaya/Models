#!/bin/bash

# Trading ML API Startup Script
# Handles initialization, validation, and server startup
# Usage: ./start-api.sh [environment] [workers]
#   environment: dev, staging, prod (default: dev)
#   workers: number of uvicorn workers (default: 4)

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT=${1:-dev}
WORKERS=${2:-4}
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/api.log"
PID_FILE="${LOG_DIR}/api.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create log directory
mkdir -p "$LOG_DIR"

log_info "Starting Trading ML API (Environment: $ENVIRONMENT, Workers: $WORKERS)"

# Activate virtual environment if exists (do this first so python points to venv)
if [ -d "${SCRIPT_DIR}/venv" ]; then
    log_info "Activating virtual environment..."
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# Check Python version (after venv activation)
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    log_error "Python 3.11+ required, found $PYTHON_VERSION"
    exit 1
fi
log_info "Python version: $PYTHON_VERSION ✓"

# Check required dependencies
log_info "Checking dependencies..."
python3 << 'EOF'
import importlib
missing = []

required = [
    ('fastapi', 'FastAPI framework'),
    ('uvicorn', 'ASGI server'),
    ('pandas', 'Data processing'),
    ('numpy', 'Numerical computing'),
    ('sklearn', 'Machine learning'),
]

for mod, desc in required:
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing.append(f"{mod} ({desc}): {e}")

# Deep learning backend: TensorFlow or PyTorch (optional; warn if neither)
dl_backends = []
for mod in ('tensorflow', 'torch'):
    try:
        importlib.import_module(mod)
        dl_backends.append(mod)
    except Exception:
        pass

if missing:
    print("✗ Missing required dependencies:\n  - " + "\n  - ".join(missing))
    raise SystemExit(1)

if not dl_backends:
    print("⚠ Deep learning backend not found (tensorflow or torch). Proceeding without DL.")

print("✓ Dependencies check passed")
EOF

# Check .env keys
log_info "Checking environment keys (.env)..."
python3 automation/ensure_env.py || log_warn "Missing keys. Run: python automation/ensure_env.py --interactive"

# Validate environment variables
log_info "Validating environment..."
MISSING_VARS=0

if [ -f "${SCRIPT_DIR}/.env" ]; then
    log_info "Loading .env file..."
    source "${SCRIPT_DIR}/.env"
fi

# Check required variables for paper trading
if [ "$ENABLE_PAPER_TRADING" = "true" ]; then
    if [ -z "$ALPACA_API_KEY" ]; then
        log_warn "ALPACA_API_KEY not set - paper trading disabled"
        MISSING_VARS=$((MISSING_VARS + 1))
    fi
    if [ -z "$ALPACA_API_SECRET" ]; then
        log_warn "ALPACA_API_SECRET not set - paper trading disabled"
        MISSING_VARS=$((MISSING_VARS + 1))
    fi
fi

if [ $MISSING_VARS -gt 0 ]; then
    log_warn "Some optional variables not configured (non-critical)"
fi

# Create required directories
log_info "Ensuring required directories exist..."
mkdir -p "${SCRIPT_DIR}/data/cache"
mkdir -p "${SCRIPT_DIR}/data/models"
mkdir -p "${SCRIPT_DIR}/data/metrics"

# Run database migrations (if applicable)
if [ -f "${SCRIPT_DIR}/migrations/migrate.py" ]; then
    log_info "Running database migrations..."
    python3 "${SCRIPT_DIR}/migrations/migrate.py" || log_warn "Migrations failed (non-critical)"
fi

# Pre-flight validation
log_info "Running pre-flight validation..."
python3 << EOF
import sys
sys.path.insert(0, '${SCRIPT_DIR}')

try:
    # Test imports
    from api.main import app
    from core.backtesting import BacktestEngine, SimpleMLPredictor
    from models.ml.advanced_trading import EnsemblePredictor
    print("✓ All imports successful")
    
    # Check if models directory is writable
    import os
    test_file = '${SCRIPT_DIR}/data/models/.write_test'
    open(test_file, 'w').close()
    os.remove(test_file)
    print("✓ Write permissions verified")
    
except Exception as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    log_error "Pre-flight validation failed"
    exit 1
fi

# Kill any existing processes on port 8000
log_info "Checking for processes on port 8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t > /dev/null; then
    log_warn "Process already running on port 8000 - stopping..."
    lsof -Pi :8000 -sTCP:LISTEN -t | xargs kill -9 || true
    sleep 2
fi

# Environment-specific configuration
case "$ENVIRONMENT" in
    dev)
        log_info "Development mode - using reload"
        UVICORN_ARGS="--reload --log-level debug"
        ;;
    staging)
        log_info "Staging mode"
        UVICORN_ARGS="--log-level info"
        ;;
    prod)
        log_info "Production mode"
        UVICORN_ARGS="--log-level info --access-log"
        ;;
    *)
        log_error "Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Start API server
log_info "Starting API server..."
log_info "Server will be available at http://localhost:8000"
log_info "Logs will be written to $LOG_FILE"

# Run uvicorn
python3 -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers $WORKERS \
    $UVICORN_ARGS \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_info "API server stopped cleanly"
else
    log_error "API server stopped with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
