#!/bin/bash
# Build script for C++ quantitative finance library

echo "=================================="
echo "Building C++ Quant Library"
echo "=================================="

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    echo "✓ Found $1"
    return 0
}

echo ""
echo "Checking dependencies..."
check_command cmake || exit 1
check_command g++ || check_command clang++ || exit 1
check_command python3 || exit 1

echo ""
echo "Installing Python build dependencies..."
python3 -m pip install pybind11 numpy --quiet || {
    echo "Error: Failed to install Python dependencies"
    exit 1
}
echo "✓ Python dependencies installed"

echo ""
echo "Building C++ library..."
python3 setup_cpp.py build_ext --inplace || {
    echo "Error: Build failed"
    exit 1
}

echo ""
echo "=================================="
echo "✓ Build successful!"
echo "=================================="
echo ""
echo "The C++ quantitative finance library is now available."
echo "Import it in Python with: from quant_accelerated import *"
echo ""
