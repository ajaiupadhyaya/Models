#!/usr/bin/env bash
# Create a working venv and install dependencies.
# Use this if "externally-managed-environment" appears when using Homebrew Python.

set -e
cd "$(dirname "$0")"

echo "=== Models project: venv setup ==="

# Option 1: Use uv if available (no EXTERNALLY-MANAGED issue)
if command -v uv &>/dev/null; then
  echo "Using uv..."
  rm -rf venv
  uv venv
  source venv/bin/activate
  uv pip install -r requirements.txt
  echo "Done. Activate with: source venv/bin/activate"
  exit 0
fi

# Option 2: Use pyenv Python if available (venv from pyenv has no EXTERNALLY-MANAGED)
if command -v pyenv &>/dev/null; then
  eval "$(pyenv init - 2>/dev/null)" || true
  PYENV_PY=$(pyenv which python3 2>/dev/null || true)
  if [ -n "$PYENV_PY" ] && [ -x "$PYENV_PY" ]; then
    echo "Using pyenv Python: $PYENV_PY"
    rm -rf venv
    "$PYENV_PY" -m venv venv
    source venv/bin/activate
    python -m pip install -r requirements.txt
    echo "Done. Activate with: source venv/bin/activate"
    exit 0
  fi
fi

# Option 3: Try system python3 -m venv (may still fail on Homebrew Python)
echo "Using $(which python3)..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate
if python -m pip install -r requirements.txt 2>/dev/null; then
  echo "Done. Activate with: source venv/bin/activate"
  exit 0
fi

echo ""
echo "=== Homebrew Python detected: venv still sees EXTERNALLY-MANAGED. ==="
echo "Install one of these, then re-run this script:"
echo ""
echo "  Option A - uv (recommended):"
echo "    brew install uv"
echo ""
echo "  Option B - pyenv + Python:"
echo "    brew install pyenv"
echo "    pyenv install 3.11.0"
echo "    pyenv local 3.11.0"
echo ""
exit 1
