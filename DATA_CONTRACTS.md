# Data Contracts

Required columns and behavior for missing or invalid data across APIs and engines.

## Backtest

- **Required columns**: `Open`, `High`, `Low`, `Close`, `Volume`. Index should be time-like (e.g. `DatetimeIndex`).
- **Missing data**: If `Close` is missing or the DataFrame is empty, the sample-data and backtest run endpoints return **404** (no price data) or **400** (insufficient data).
- **NaN/Inf**: Engines use `Close` for execution; NaN in `Close` can produce fewer valid bars. Behavior: bars with NaN are typically skipped or cause incomplete equity series. For reproducible backtests, use clean data (dropna or forward-fill as appropriate).
- **Minimum bars**: No hard minimum in the engine; very short series produce few or no trades.

## Risk (GET /api/v1/risk/metrics/{ticker})

- **Required columns**: `Close` (price series). Other columns are not used for risk metrics.
- **Missing data**: No DataFrame or empty DataFrame → **404** "No price data for {ticker}".
- **No Close column**: → **404**.
- **Insufficient data**: After `pct_change().dropna()`, fewer than 20 returns → **400** "Insufficient data for risk metrics".
- **NaN/Inf in Close**: `calculate_returns` drops NaN; effective number of returns may drop below 20 → **400**. If enough valid returns remain, metrics are computed on the valid subset; no explicit NaN/Inf check beyond that.

## Company / fundamental analysis

- **Data sources**: Company analysis uses DataFetcher (price), plus fundamental data (income statement, balance sheet, cash flow) when available.
- **Missing price**: Same as risk: no or empty price data leads to error or empty risk subsection.
- **Missing fundamental data**: Ratios and valuation may be partial or missing; endpoints return what is available or an error message.

## Data fetcher (core/data_fetcher.py)

- **Stock data**: Returns DataFrame with OHLCV or None on failure. Callers should check `data is None or data.empty` and for `"Close" in data.columns` before use.
- **Reproducibility**: Use fixed seeds in tests. Point-in-time and survivorship bias are documented goals in ARCHITECTURE; production data should be point-in-time where possible.

## Summary

| Use case     | Required columns | No/empty data | No Close | Insufficient returns | NaN/Inf in Close      |
|-------------|------------------|---------------|----------|----------------------|------------------------|
| Backtest    | OHLCV            | 404/400       | 404      | 400                  | Fewer bars / skip      |
| Risk API    | Close            | 404           | 404      | 400                  | Fewer returns → 400 or computed |
| Company     | Close + optional | error/partial | error    | -                    | Same as risk for price |
