# Terminal Workflows

Step-by-step example workflows for the financial analysis terminal. All flows assume the API is running (e.g. `uvicorn api.main:app --reload` on port 8000) and the frontend is served with proxy to the API (e.g. `npm run dev` in `frontend/`).

---

## 1. Daily macro snapshot

**Goal:** View key macroeconomic indicators (FRED-based).

1. Focus the command bar: press **/** or **Cmd+K** (Mac) / **Ctrl+K** (Windows).
2. Type **ECO** and press **Enter**.
3. The **Economic** module opens and loads macro series from `/api/v1/data/macro`.
4. If data fails to load, use the **Retry** button; the client retries up to 2 times on 5xx or network errors.
5. Ensure `FRED_API_KEY` is set in the environment (or in `.env`) so the API can fetch FRED series.

**Optional:** Use **Alt+Left** / **Alt+Right** to move to the previous/next module without typing a command.

---

## 2. Single-asset deep dive

**Goal:** Analyze one symbol across fundamentals, chart, technicals, and news.

1. **Set symbol**
   - Focus command bar and type a ticker (e.g. **AAPL**) then **Enter**, or  
   - Click a symbol in the **Watchlist** (left panel). The **Primary Instrument** chart loads for that symbol.

2. **Fundamental analysis**
   - Type **FA** or **FA AAPL** and **Enter** to open the Fundamental module.
   - View ratios, DCF valuation, and risk metrics from `/api/v1/company/analyze/{ticker}`.
   - On error, use **Retry** to refetch.

3. **Chart and technicals**
   - Type **GP** or **GP AAPL** (or **FLDS**) to open the Primary/Technical view.
   - Use timeframe buttons (1D, 5D, 1M, 3M) to change the period.
   - Toggle **SMA 20**, **SMA 50**, **RSI**, **MACD**, or **Bollinger** in the Technical panel to overlay indicators on the chart.
   - If price data fails, use the **Retry** button above the chart.

4. **News / summary**
   - Type **N** or **N AAPL** to open the News module.
   - Market summary for the primary symbol is loaded from `/api/v1/ai/market-summary`; use **Retry** if the request fails.

---

## 3. Factor research and strategy backtest

**Goal:** Run a backtest for the current (or last backtest) symbol and view metrics.

1. Open the **Quant** module: type **BACKTEST** or **PORT** and then switch to Quant with **Alt+Right**, or type a backtest-related command if wired.
2. The panel loads the list of models from `/api/v1/models`. If that request fails, use **Retry** next to the Models line.
3. Click **Run backtest ({symbol})** to POST to `/api/v1/backtest/run` with default parameters (e.g. 1-year window, institutional engine if configured).
4. After the run, view **Sharpe**, **Max drawdown**, and **Total return** in the same panel.
5. Backtest uses the institutional engine by default (slippage, transaction costs, market impact) when `BACKTEST_USE_INSTITUTIONAL_DEFAULT` is true in config.

---

## 4. Portfolio and risk

**Goal:** See ML signals, active models, and risk metrics (VaR, CVaR, volatility, max drawdown, Sharpe) for the primary symbol.

1. Type **PORT** and **Enter** to open the **Portfolio & Strategies** module.
2. The panel loads:
   - **Quick predict** (ML signal) for the primary symbol from `/api/v1/predictions/quick-predict` (refreshes on an interval).
   - **Dashboard** (active models, recent predictions) from `/api/v1/monitoring/dashboard`. On failure, use **Retry**.
   - **Risk** subsection from `/api/v1/risk/metrics/{ticker}?period=1y`. If risk fails, use **Retry** in the Risk section.
3. Risk metrics shown: VaR 95%, VaR 99%, CVaR 95%, Vol (ann.), Max DD, Sharpe.

---

## 5. Screening and discovery

**Goal:** View sectors or screener placeholders.

1. Type **SCREEN** (or the screening command) and **Enter** to open the **Screening & discovery** module.
2. The panel fetches `/api/v1/company/sectors`. Use **Retry** if the request fails.
3. Results are shown as a table; multi-factor screener can be wired to the same or additional endpoints.

---

## 6. AI-assisted research

**Goal:** Ask a question and see the answer in the AI Assistant panel.

1. The **AI Assistant** is in the right panel; it uses analysis and prediction APIs.
2. Type **AI** plus your query in the command bar, or use the dedicated AI input if available.
3. The assistant calls endpoints such as `/api/v1/ai/stock-analysis/{symbol}` and displays the result in the panel.

---

## 7. Workspace and layout

**Goal:** Save and restore layout, active module, and primary symbol.

1. Workspaces are persisted in `localStorage` by name.
2. Use the **WORKSPACE** command (or UI control) to switch or create workspaces.
3. Layout, active module, and primary symbol are restored when you return to a workspace.

---

## 8. Chart export (optional)

- Primary Instrument chart (D3 candlestick + volume + indicators) can be extended with an "Export chart as PNG/SVG" button; document in UI when implemented.

---

## 9. Help and navigation

- **?** or **HELP** in the command bar: open the help overlay with available commands and keyboard shortcuts.
- **/** or **Cmd+K** / **Ctrl+K**: Focus command bar.
- **Alt+Left** / **Alt+Right**: Previous / next module.
- **Esc**: Clear command bar and blur.
- **Enter**: Submit command or ticker.
- **Arrow Up/Down**: Command history when the command bar is focused.

---

## 10. Error handling and retry

- Data-fetching panels use a shared **retry** pattern: up to 2 automatic retries on 5xx or network errors, with exponential backoff.
- Each panel shows a **Retry** button when the request fails so you can manually refetch without leaving the view.
- Ensure the API is running on the configured port (default 8000) and that the frontend proxy targets it (e.g. Vite proxy in `frontend/vite.config.ts`).
