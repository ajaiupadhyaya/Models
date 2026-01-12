# Installation Guide

## Step 1: Check Python Version

```bash
python3 --version
```

**Recommended: Python 3.12 or 3.11** for full package compatibility.

You need Python 3.8 or higher, but we strongly recommend Python 3.12 for:
- Full package ecosystem support (including cvxpy, PyPortfolioOpt)
- Production stability
- Better compatibility with all financial libraries

## Step 2: Create Virtual Environment (Recommended)

**If you have Python 3.12 (recommended):**
```bash
cd /Users/ajaiupadhyaya/Documents/Models
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**If using default Python 3.8+:**
```bash
cd /Users/ajaiupadhyaya/Documents/Models
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Note:** Python 3.12 ensures all packages install correctly, including cvxpy and PyPortfolioOpt.

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Some packages may require additional system dependencies:
- On macOS: May need Xcode Command Line Tools
- On Linux: May need build-essential, python3-dev

If you encounter issues with specific packages:

### QuantLib (Optional)
QuantLib can be tricky to install. If it fails, you can skip it - most models don't require it:
```bash
pip install -r requirements.txt --ignore-installed quantlib
```

### Alternative: Install Core Packages Only
If you want to start with just the essentials:
```bash
pip install numpy pandas scipy matplotlib plotly jupyter yfinance requests python-dotenv
```

## Step 4: Configure API Keys (Optional)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   FRED_API_KEY=your_key_here
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

   Get free API keys:
   - **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html
   - **Alpha Vantage**: https://www.alphavantage.co/support/#api-key

## Step 5: Verify Installation

```bash
python3 quick_start.py
```

You should see all tests passing (API key tests may show warnings if keys aren't set, which is fine).

## Step 6: Start Jupyter Lab

```bash
jupyter lab
```

Then navigate to the `notebooks/` directory to see examples.

## Troubleshooting

### Issue: "ModuleNotFoundError"
- Make sure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

### Issue: "Permission denied"
- Use `pip install --user` or activate virtual environment

### Issue: "Failed building wheel"
- Install build tools:
  - macOS: `xcode-select --install`
  - Ubuntu/Debian: `sudo apt-get install build-essential python3-dev`
  - Windows: Install Visual C++ Build Tools

### Issue: QuantLib installation fails
- This is optional. Most models work without it.
- Skip it or install separately: `conda install -c conda-forge quantlib-python`

### Issue: Python environment is broken
- Create a fresh virtual environment:
  ```bash
  rm -rf venv
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## Next Steps

1. Read `README.md` for overview
2. Check `USAGE.md` for usage examples
3. Open `notebooks/01_getting_started.ipynb` to begin
