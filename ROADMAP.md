# Bloomberg Terminal Implementation Roadmap

## Vision Statement
A fast, responsive, intuitive, and cutting-edge local Bloomberg Terminal that serves as the single point of control for monitoring and observing an automated system of charting, analysis, and trading. Everything should be "plugged in" to be fully automated and flow seamlessly.

---

## Current Status ✅

### Completed Foundation
- ✅ FastAPI backend with clean router organization
- ✅ Enhanced orchestrator (single canonical engine)
- ✅ React + D3 frontend shell with basic layout
- ✅ LLM provider abstraction layer
- ✅ Legacy Dash/Plotly deprecation
- ✅ Core API endpoints (market data, backtesting, AI, orchestrator, WebSocket)

### What's Working
- Backend API server (`/api/v1/*` endpoints)
- Basic React terminal shell with three-panel layout
- D3 watchlist visualization (MarketOverview)
- D3 candlestick chart (PrimaryInstrument - placeholder endpoint)
- AI assistant panel (basic integration)
- WebSocket infrastructure for real-time updates

---

## Phase 1: Core Data Integration (Priority: HIGH) 🔴

**Goal:** Wire up real market data endpoints to replace placeholders and enable live data flow.

### 1.1 Market Data Endpoint for Primary Instrument
**Current:** `PrimaryInstrument.tsx` uses placeholder `/api/v1/backtest/sample-data`

**Action Items:**
- [ ] Create `/api/v1/market/data/{symbol}` endpoint that returns OHLCV candles
  - Use `core/data_fetcher.py` → `get_stock_data(symbol, period)`
  - Return JSON: `{dates: [], open: [], high: [], low: [], close: [], volume: []}`
  - Support query params: `period` (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
- [ ] Update `PrimaryInstrument.tsx` to fetch from `/api/v1/market/data/{symbol}`
- [ ] Add timeframe selector (1D, 5D, 1M, 3M, 6M, 1Y) in UI
- [ ] Handle loading states and errors gracefully

**Files to Modify:**
- `api/main.py` - Add new router or extend existing
- `frontend/src/terminal/panels/PrimaryInstrument.tsx` - Update fetch URL
- Consider: `api/market_data_api.py` (new file) or extend `api/company_analysis_api.py`

### 1.2 Real-Time Price Updates via WebSocket
**Current:** WebSocket infrastructure exists but not connected to frontend

**Action Items:**
- [ ] Create React hook `useWebSocket.ts`:
  ```typescript
  export function useWebSocket<T>(url: string, onMessage: (data: T) => void)
  ```
- [ ] Connect `MarketOverview.tsx` to `ws://localhost:8000/api/v1/ws/prices/{symbol}`
- [ ] Update watchlist bars in real-time as prices stream
- [ ] Add connection status indicator (green/red dot)
- [ ] Handle reconnection logic

**Files to Create:**
- `frontend/src/hooks/useWebSocket.ts`
- `frontend/src/hooks/usePriceStream.ts` (specialized for prices)

**Files to Modify:**
- `frontend/src/terminal/panels/MarketOverview.tsx` - Add WebSocket subscription

### 1.3 Watchlist Management
**Current:** Hardcoded symbols in MarketOverview

**Action Items:**
- [ ] Add watchlist state management (localStorage or backend)
- [ ] Create `WatchlistManager.tsx` component:
  - Add symbol input with autocomplete (use `/api/v1/company/search?query={q}`)
  - Remove symbol button
  - Drag-to-reorder (optional)
- [ ] Persist watchlist to localStorage or `/api/v1/user/watchlist` (if you add user management)
- [ ] Update MarketOverview to read from watchlist state

**Files to Create:**
- `frontend/src/terminal/components/WatchlistManager.tsx`
- `frontend/src/hooks/useWatchlist.ts`

**Files to Modify:**
- `frontend/src/terminal/panels/MarketOverview.tsx` - Use watchlist hook

---

## Phase 2: Advanced Charting with D3 (Priority: HIGH) 🔴

**Goal:** Build production-grade D3 charts that match Bloomberg's visual quality and interactivity.

### 2.1 Enhanced Candlestick Chart
**Current:** Basic D3 candlestick with minimal features

**Action Items:**
- [ ] Add technical indicators overlay:
  - Moving averages (MA20, MA50, MA200) - toggleable
  - Bollinger Bands - toggleable
  - Volume bars below price chart
- [ ] Add zoom/pan with D3 brush
- [ ] Add crosshair cursor showing OHLCV on hover
- [ ] Add time range selector (1D, 5D, 1M, 3M, 6M, 1Y, MAX)
- [ ] Add drawing tools (trend lines, support/resistance) - optional
- [ ] Responsive resizing

**Files to Create:**
- `frontend/src/terminal/charts/CandlestickChart.tsx` (extract from PrimaryInstrument)
- `frontend/src/terminal/charts/indicators.ts` (MA, Bollinger calculations)
- `frontend/src/terminal/charts/ChartControls.tsx` (toggle indicators, timeframes)

**Files to Modify:**
- `frontend/src/terminal/panels/PrimaryInstrument.tsx` - Use new chart component

### 2.2 Portfolio Equity Curve & Drawdown Chart
**Current:** PortfolioPanel is a placeholder

**Action Items:**
- [ ] Create `/api/v1/portfolio/equity-curve` endpoint:
  - Returns equity curve from orchestrator or paper trading
  - Format: `{dates: [], equity: [], drawdown: []}`
- [ ] Build D3 equity curve chart:
  - Line chart showing portfolio value over time
  - Underlay drawdown area (red when negative)
  - Show key metrics: total return, Sharpe, max drawdown
- [ ] Add strategy comparison (if multiple strategies running)
- [ ] Add performance attribution breakdown (by symbol/sector)

**Files to Create:**
- `frontend/src/terminal/charts/EquityCurveChart.tsx`
- `frontend/src/terminal/charts/DrawdownChart.tsx`
- `frontend/src/terminal/panels/PortfolioPanel.tsx` - Full implementation

**Files to Modify:**
- `api/orchestrator_api.py` or create `api/portfolio_api.py` - Add equity curve endpoint

### 2.3 Market Heatmap (Sector/Asset Class)
**Action Items:**
- [ ] Create `/api/v1/market/heatmap` endpoint:
  - Returns sector/asset performance matrix
  - Format: `{sectors: [], symbols: [], returns: [], colors: []}`
- [ ] Build D3 treemap or heatmap visualization:
  - Color-coded by performance (green/red)
  - Size by market cap or volume
  - Click to drill into sector/symbol
- [ ] Add to MarketOverview or new panel

**Files to Create:**
- `frontend/src/terminal/charts/MarketHeatmap.tsx`
- `api/market_data_api.py` - Add heatmap endpoint

### 2.4 Strategy Performance Comparison
**Action Items:**
- [ ] Use existing `/api/v1/backtest/compare` endpoint
- [ ] Build D3 multi-line chart comparing strategies:
  - Equity curves overlaid
  - Sharpe ratio comparison bar chart
  - Drawdown comparison
- [ ] Add strategy selector dropdown
- [ ] Show metrics table (Sharpe, return, max DD, win rate)

**Files to Create:**
- `frontend/src/terminal/charts/StrategyComparison.tsx`
- `frontend/src/terminal/panels/StrategyPanel.tsx`

---

## Phase 3: Real-Time Automation Visibility (Priority: MEDIUM) 🟡

**Goal:** Make the UI reflect live automation state (running strategies, data updates, alerts).

### 3.1 Orchestrator Status Dashboard
**Current:** `/api/v1/orchestrator/status` exists but not displayed

**Action Items:**
- [ ] Create `OrchestratorStatus.tsx` component:
  - Show running strategies (from orchestrator status)
  - Model counts (ensemble, LSTM, RL)
  - Last retrain timestamps
  - Market regime indicator (from enhanced orchestrator)
  - Active positions count
- [ ] Poll `/api/v1/orchestrator/status` every 30 seconds
- [ ] Add visual indicators (green = active, yellow = warning, red = error)
- [ ] Show scheduler status (next run time)

**Files to Create:**
- `frontend/src/terminal/panels/OrchestratorStatus.tsx`
- `frontend/src/hooks/useOrchestratorStatus.ts`

**Files to Modify:**
- `frontend/src/terminal/TerminalShell.tsx` - Add status panel (maybe top bar or sidebar)

### 3.2 Live Trading Signals Panel
**Current:** Signals exist in orchestrator but not streamed to UI

**Action Items:**
- [ ] Create `/api/v1/orchestrator/signals/stream` WebSocket endpoint:
  - Streams new signals as they're generated
  - Format: `{symbol, action, confidence, price, reasoning, timestamp}`
- [ ] Build `TradingSignalsPanel.tsx`:
  - Real-time list of signals (BUY/SELL/HOLD)
  - Color-coded by action (green/red/gray)
  - Confidence bars
  - Click to view full analysis
- [ ] Add sound/visual notification for high-confidence signals (optional)
- [ ] Filter by symbol, action, confidence threshold

**Files to Create:**
- `frontend/src/terminal/panels/TradingSignalsPanel.tsx`
- `frontend/src/hooks/useSignalStream.ts`

**Files to Modify:**
- `api/orchestrator_api.py` - Add WebSocket signal streaming
- `frontend/src/terminal/TerminalShell.tsx` - Add signals panel

### 3.3 Alert & Notification Center
**Current:** Alerting system exists in backend but not exposed to UI

**Action Items:**
- [ ] Create `/api/v1/alerts` endpoint (or extend monitoring API):
  - Returns recent alerts from `core/alerting_system.py`
  - Format: `{alerts: [{severity, title, message, symbol, timestamp}], unread_count}`
- [ ] Build `AlertCenter.tsx` component:
  - Notification bell icon with badge (unread count)
  - Dropdown/popover showing recent alerts
  - Severity indicators (info/warning/critical)
  - Mark as read functionality
- [ ] Add WebSocket for real-time alerts: `ws://localhost:8000/api/v1/ws/alerts`
- [ ] Toast notifications for critical alerts

**Files to Create:**
- `frontend/src/terminal/components/AlertCenter.tsx`
- `frontend/src/hooks/useAlerts.ts`

**Files to Modify:**
- `api/monitoring.py` - Add alerts endpoint (or create `api/alerts_api.py`)
- `api/websocket_api.py` - Add alert streaming

### 3.4 Data Refresh Status
**Action Items:**
- [ ] Show last data update timestamp in header
- [ ] Add refresh button (manual trigger)
- [ ] Show data source status (yfinance, FRED, Alpha Vantage)
- [ ] Display any data fetch errors

**Files to Modify:**
- `frontend/src/terminal/TerminalShell.tsx` - Add status to header
- `api/monitoring.py` - Add data refresh status endpoint

---

## Phase 4: Enhanced AI Assistant (Priority: MEDIUM) 🟡

**Goal:** Make the AI assistant context-aware and deeply integrated with the terminal.

### 4.1 Context-Aware AI Chat
**Current:** Basic AI assistant that takes symbol input

**Action Items:**
- [ ] Enhance AI assistant to inject context:
  - Currently selected symbol
  - Current price and recent performance
  - Active positions
  - Recent signals
  - Market regime
- [ ] Add conversation history (persist in localStorage or backend)
- [ ] Support natural language queries:
  - "What's the best strategy for AAPL?"
  - "Why did the orchestrator generate a SELL signal for TSLA?"
  - "Compare AAPL and MSFT"
- [ ] Add quick actions:
  - "Run backtest for AAPL"
  - "Generate investor report"
  - "Analyze risk for my portfolio"

**Files to Modify:**
- `frontend/src/terminal/panels/AiAssistantPanel.tsx` - Add context injection
- `api/ai_analysis_api.py` - Add `/chat` endpoint with context support
- Consider: `api/ai_chat_api.py` (new file for conversational AI)

### 4.2 AI-Generated Insights Panel
**Action Items:**
- [ ] Create `/api/v1/ai/insights/{symbol}` endpoint:
  - Returns AI-generated daily/weekly insights
  - Combines technical, fundamental, sentiment analysis
- [ ] Build `InsightsPanel.tsx`:
  - Display key insights in card format
  - Auto-refresh daily
  - Link to full analysis

**Files to Create:**
- `frontend/src/terminal/panels/InsightsPanel.tsx`
- `api/ai_analysis_api.py` - Add insights endpoint

### 4.3 Report Generation Integration
**Current:** Investor reports API exists but not accessible from UI

**Action Items:**
- [ ] Add "Generate Report" button in AI assistant or symbol context menu
- [ ] Use `/api/v1/investor-reports/generate` endpoint
- [ ] Show report preview in modal or new tab
- [ ] Download as PDF/Markdown

**Files to Modify:**
- `frontend/src/terminal/panels/AiAssistantPanel.tsx` - Add report generation
- Or create: `frontend/src/terminal/components/ReportGenerator.tsx`

---

## Phase 5: Search & Navigation (Priority: MEDIUM) 🟡

**Goal:** Bloomberg-style search and quick navigation.

### 5.1 Global Search Bar
**Action Items:**
- [ ] Add search bar to terminal header
- [ ] Integrate `/api/v1/company/search?query={q}`:
  - Autocomplete dropdown
  - Show company name, ticker, sector
  - Click to navigate to symbol
- [ ] Support ticker shortcuts (type "AAPL" → jump to chart)
- [ ] Search history (recent searches)

**Files to Create:**
- `frontend/src/terminal/components/GlobalSearch.tsx`
- `frontend/src/hooks/useCompanySearch.ts`

**Files to Modify:**
- `frontend/src/terminal/TerminalShell.tsx` - Add search to header

### 5.2 Symbol Navigation & Context Menu
**Action Items:**
- [ ] Right-click on symbol → context menu:
  - "View Chart"
  - "Run Analysis"
  - "Add to Watchlist"
  - "Generate Report"
  - "Backtest Strategy"
- [ ] Keyboard shortcuts:
  - `Cmd/Ctrl + K` → Open search
  - `Cmd/Ctrl + P` → Quick symbol jump
  - Arrow keys → Navigate watchlist

**Files to Create:**
- `frontend/src/terminal/components/SymbolContextMenu.tsx`
- `frontend/src/hooks/useKeyboardShortcuts.ts`

---

## Phase 6: Performance & Polish (Priority: LOW) 🟢

**Goal:** Ensure the terminal is fast, responsive, and handles edge cases gracefully.

### 6.1 D3 Performance Optimization
**Action Items:**
- [ ] Implement data downsampling for large timeframes (1Y+)
- [ ] Use D3's `d3.bisector` for efficient data lookups
- [ ] Virtual scrolling for long watchlists
- [ ] Debounce/throttle WebSocket updates
- [ ] Lazy load chart components

**Files to Modify:**
- All chart components in `frontend/src/terminal/charts/`
- `frontend/src/hooks/useWebSocket.ts` - Add throttling

### 6.2 Error Handling & Loading States
**Action Items:**
- [ ] Add error boundaries in React
- [ ] Show loading skeletons for all async operations
- [ ] Graceful degradation when APIs are unavailable
- [ ] Retry logic for failed API calls
- [ ] User-friendly error messages

**Files to Create:**
- `frontend/src/components/ErrorBoundary.tsx`
- `frontend/src/components/LoadingSkeleton.tsx`
- `frontend/src/utils/errorHandler.ts`

### 6.3 Responsive Design
**Action Items:**
- [ ] Test on different screen sizes
- [ ] Collapsible panels for mobile/tablet
- [ ] Touch-friendly controls
- [ ] Optimize for ultrawide monitors (Bloomberg-style)

**Files to Modify:**
- `frontend/src/styles.css` - Add responsive breakpoints
- All panel components - Add collapse/expand

### 6.4 Accessibility
**Action Items:**
- [ ] ARIA labels for all interactive elements
- [ ] Keyboard navigation support
- [ ] Screen reader compatibility
- [ ] High contrast mode option

---

## Phase 7: Advanced Features (Priority: LOW) 🟢

**Goal:** Add Bloomberg Terminal-level advanced features.

### 7.1 Multi-Symbol Comparison
**Action Items:**
- [ ] Allow selecting multiple symbols
- [ ] Overlay charts (normalized to percentage change)
- [ ] Correlation matrix visualization
- [ ] Performance comparison table

**Files to Create:**
- `frontend/src/terminal/panels/ComparisonPanel.tsx`
- `frontend/src/terminal/charts/MultiSymbolChart.tsx`

### 7.2 Custom Indicators & Studies
**Action Items:**
- [ ] Add indicator selector (RSI, MACD, Stochastic, etc.)
- [ ] Custom indicator builder (user-defined formulas)
- [ ] Save indicator presets

**Files to Create:**
- `frontend/src/terminal/components/IndicatorSelector.tsx`
- `api/indicators_api.py` - Calculate indicators server-side

### 7.3 Portfolio Analytics Dashboard
**Action Items:**
- [ ] Full portfolio view with positions
- [ ] P&L breakdown by symbol/sector
- [ ] Risk metrics (VaR, CVaR, beta)
- [ ] Exposure analysis (sector, region, asset class)
- [ ] Rebalancing suggestions

**Files to Create:**
- `frontend/src/terminal/panels/PortfolioAnalytics.tsx`
- `api/portfolio_api.py` - Portfolio analytics endpoints

### 7.4 Backtesting UI
**Action Items:**
- [ ] Visual backtest configuration form
- [ ] Run backtest from UI
- [ ] Display results with charts
- [ ] Parameter optimization interface

**Files to Create:**
- `frontend/src/terminal/panels/BacktestPanel.tsx`
- Use existing `/api/v1/backtest/run` endpoint

---

## Implementation Priority Matrix

```
HIGH PRIORITY (Do First):
├── Phase 1: Core Data Integration
│   ├── 1.1 Market Data Endpoint ✅ Critical
│   ├── 1.2 Real-Time WebSocket ✅ Critical
│   └── 1.3 Watchlist Management ✅ Important
└── Phase 2: Advanced Charting
    ├── 2.1 Enhanced Candlestick ✅ Critical
    ├── 2.2 Portfolio Equity Curve ✅ Critical
    └── 2.3 Market Heatmap ✅ Nice-to-have

MEDIUM PRIORITY (Do Next):
├── Phase 3: Automation Visibility
│   ├── 3.1 Orchestrator Status ✅ Important
│   ├── 3.2 Live Signals Panel ✅ Important
│   └── 3.3 Alert Center ✅ Important
└── Phase 4: Enhanced AI Assistant
    ├── 4.1 Context-Aware Chat ✅ Important
    └── 4.2 Insights Panel ✅ Nice-to-have

LOW PRIORITY (Polish Later):
├── Phase 5: Search & Navigation
├── Phase 6: Performance & Polish
└── Phase 7: Advanced Features
```

---

## Quick Wins (Can Do Immediately)

1. **Wire PrimaryInstrument to real data** (30 min)
   - Create `/api/v1/market/data/{symbol}` endpoint
   - Update `PrimaryInstrument.tsx` fetch URL

2. **Add WebSocket price updates** (1 hour)
   - Create `useWebSocket.ts` hook
   - Connect MarketOverview to price stream

3. **Complete PortfolioPanel** (2 hours)
   - Create equity curve endpoint
   - Build D3 equity/drawdown chart

4. **Add orchestrator status display** (1 hour)
   - Poll `/api/v1/orchestrator/status`
   - Show in header or sidebar

---

## Technical Debt & Considerations

### Backend
- [ ] Consider adding Redis for WebSocket message queuing (if scaling)
- [ ] Add rate limiting to prevent API abuse
- [ ] Add request caching for expensive operations (market data, analysis)
- [ ] Consider GraphQL for complex queries (optional)

### Frontend
- [ ] Add state management (Zustand/Redux) if complexity grows
- [ ] Consider code splitting for faster initial load
- [ ] Add E2E tests (Playwright/Cypress) for critical flows
- [ ] Set up CI/CD for frontend builds

### Data
- [ ] Consider WebSocket connection pooling for multiple symbols
- [ ] Add data validation on API responses
- [ ] Implement request deduplication (same symbol requested multiple times)

---

## Success Metrics

- ✅ **Speed**: Page load < 2s, chart render < 500ms
- ✅ **Responsiveness**: UI updates within 100ms of data arrival
- ✅ **Reliability**: 99.9% uptime, graceful error handling
- ✅ **Intuitiveness**: New user can navigate without documentation
- ✅ **Automation Visibility**: All running strategies visible at a glance

---

## Next Immediate Steps

1. **Start with Phase 1.1** - Create market data endpoint and wire PrimaryInstrument
2. **Then Phase 1.2** - Add WebSocket price streaming to watchlist
3. **Then Phase 2.2** - Complete PortfolioPanel with equity curve
4. **Then Phase 3.1** - Add orchestrator status display

This roadmap ensures the terminal becomes a fully functional, Bloomberg-style monitoring and control center for your automated trading system.
