# Developer Quick Reference Guide

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- Virtual environment (venv)

### Setup
```bash
# Backend
pip install -r requirements.txt requirements-api.txt
cd api && uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Code Patterns

### Adding a New API Endpoint

**Backend (`api/`):**
```python
# api/example_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/example", tags=["Example"])

class ExampleRequest(BaseModel):
    symbol: str

@router.get("/{symbol}")
async def get_example(symbol: str):
    try:
        # Your logic here
        return {"symbol": symbol, "data": "..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Then in api/main.py:
from api.example_api import router as example_router
app.include_router(example_router)
```

**Frontend (`frontend/src/`):**
```typescript
// hooks/useExample.ts
export function useExample(symbol: string) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/v1/example/${symbol}`)
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, [symbol]);

  return { data, loading };
}

// Usage in component
const { data, loading } = useExample("AAPL");
```

### Adding a WebSocket Stream

**Backend:**
```python
# api/websocket_api.py
@router.websocket("/example/{symbol}")
async def websocket_example(websocket: WebSocket, symbol: str):
    manager = get_app_state()["connection_manager"]
    await manager.connect(websocket)
    
    try:
        while True:
            # Fetch data
            data = fetch_data(symbol)
            
            # Send update
            await websocket.send_json({
                "type": "update",
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

**Frontend:**
```typescript
// hooks/useWebSocket.ts (create if doesn't exist)
export function useWebSocket<T>(url: string, onMessage: (data: T) => void) {
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000${url}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };
    
    return () => ws.close();
  }, [url]);
}

// Usage
useWebSocket("/api/v1/ws/example/AAPL", (data) => {
  setLatestData(data);
});
```

### Creating a D3 Chart Component

```typescript
// terminal/charts/ExampleChart.tsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

interface Props {
  data: Array<{ date: Date; value: number }>;
  width?: number;
  height?: number;
}

export const ExampleChart: React.FC<Props> = ({ data, width = 600, height = 400 }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || !data.length) return;

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous render
    d3.select(ref.current).selectAll("*").remove();

    const svg = d3
      .select(ref.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3
      .scaleTime()
      .domain(d3.extent(data, d => d.date) as [Date, Date])
      .range([0, innerWidth]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(data, d => d.value) as [number, number])
      .nice()
      .range([innerHeight, 0]);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x));

    g.append("g").call(d3.axisLeft(y));

    // Line
    const line = d3
      .line<{ date: Date; value: number }>()
      .x(d => x(d.date))
      .y(d => y(d.value))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#58a6ff")
      .attr("stroke-width", 2)
      .attr("d", line);

  }, [data, width, height]);

  return <div ref={ref} />;
};
```

### Adding a New Panel

```typescript
// terminal/panels/ExamplePanel.tsx
import React from "react";
import { useExample } from "../../hooks/useExample";

export const ExamplePanel: React.FC = () => {
  const { data, loading } = useExample("AAPL");

  return (
    <section className="panel panel-main">
      <div className="panel-title">Example Panel</div>
      {loading ? (
        <div className="panel-body-muted">Loading...</div>
      ) : (
        <div className="panel-body">
          {/* Your content */}
        </div>
      )}
    </section>
  );
};

// Then add to TerminalShell.tsx
import { ExamplePanel } from "./panels/ExamplePanel";
// ... in JSX
<ExamplePanel />
```

## File Structure Conventions

```
frontend/src/
├── terminal/
│   ├── TerminalShell.tsx          # Main layout
│   ├── panels/                     # Panel components
│   │   ├── MarketOverview.tsx
│   │   ├── PrimaryInstrument.tsx
│   │   └── ...
│   ├── charts/                     # D3 chart components
│   │   ├── CandlestickChart.tsx
│   │   └── ...
│   └── components/                  # Reusable UI components
│       ├── WatchlistManager.tsx
│       └── ...
├── hooks/                          # Custom React hooks
│   ├── useWebSocket.ts
│   ├── useWatchlist.ts
│   └── ...
├── utils/                          # Utility functions
│   └── formatters.ts
└── styles.css                      # Global styles

api/
├── main.py                         # FastAPI app entry
├── example_api.py                  # Domain-specific routers
└── ...

core/
├── enhanced_orchestrator.py        # Main orchestrator
├── data_fetcher.py                 # Data fetching
├── ai/                             # AI/LLM abstractions
└── ...
```

## Naming Conventions

- **Components**: PascalCase (`MarketOverview.tsx`)
- **Hooks**: camelCase starting with `use` (`useWebSocket.ts`)
- **Utils**: camelCase (`formatters.ts`)
- **API endpoints**: kebab-case (`/api/v1/market-data`)
- **Python modules**: snake_case (`market_data_api.py`)
- **Python classes**: PascalCase (`MarketDataAPI`)

## Styling Guidelines

- Use CSS variables from `styles.css` (`var(--bg)`, `var(--accent)`)
- Follow Bloomberg Terminal dark theme
- Panels: `className="panel panel-{position}"`
- Consistent spacing: 8px, 12px, 16px multiples

## Error Handling Pattern

```typescript
// Always handle errors gracefully
try {
  const res = await fetch("/api/v1/endpoint");
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  const data = await res.json();
  setData(data);
} catch (err) {
  console.error("Failed to fetch:", err);
  setError(err.message);
  // Show user-friendly error message
}
```

## Testing Checklist

Before submitting:
- [ ] Test with real API endpoints (not mocks)
- [ ] Test error cases (network failure, invalid data)
- [ ] Test loading states
- [ ] Test responsive design (mobile, tablet, desktop)
- [ ] Check browser console for errors
- [ ] Verify WebSocket reconnection works
- [ ] Test with multiple symbols simultaneously

## Common Tasks

### Adding a Symbol to Watchlist
```typescript
// Use localStorage or backend endpoint
const addToWatchlist = (symbol: string) => {
  const current = JSON.parse(localStorage.getItem("watchlist") || "[]");
  if (!current.includes(symbol)) {
    localStorage.setItem("watchlist", JSON.stringify([...current, symbol]));
  }
};
```

### Formatting Numbers
```typescript
// Use Intl.NumberFormat for consistent formatting
const formatPrice = (price: number) => 
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2
  }).format(price);

const formatPercent = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
```

### Date Formatting
```typescript
// Use date-fns or native Intl
import { format } from "date-fns";

const formatDate = (date: Date) => format(date, "MMM dd, yyyy");
const formatTime = (date: Date) => format(date, "HH:mm:ss");
```

## Debugging Tips

1. **Backend**: Check FastAPI docs at `http://localhost:8000/docs`
2. **Frontend**: Use React DevTools and D3 inspector
3. **WebSocket**: Check Network tab → WS filter
4. **API Calls**: Check Network tab → XHR filter
5. **State**: Use React DevTools → Components → Inspect state

## Performance Tips

1. **Memoize expensive calculations**: `useMemo(() => expensiveCalc(data), [data])`
2. **Debounce user input**: Use `lodash.debounce` or custom hook
3. **Lazy load charts**: Only render when panel is visible
4. **Limit WebSocket subscriptions**: Unsubscribe when component unmounts
5. **Cache API responses**: Use React Query or SWR

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-heatmap-chart

# Make changes, commit
git add .
git commit -m "feat: add market heatmap chart"

# Push and create PR
git push origin feature/add-heatmap-chart
```

## Quick Reference: Key Endpoints

```typescript
// Market Data
GET /api/v1/market/data/{symbol}?period=1mo

// Company Analysis
GET /api/v1/company/analyze/{ticker}
GET /api/v1/company/search?query={q}

// Orchestrator
GET /api/v1/orchestrator/status
GET /api/v1/orchestrator/signals?symbol={sym}&limit=10

// Backtesting
POST /api/v1/backtest/run
GET /api/v1/backtest/metrics

// AI
GET /api/v1/ai/stock-analysis/{symbol}
POST /api/v1/ai/chat

// WebSocket
ws://localhost:8000/api/v1/ws/prices/{symbol}
ws://localhost:8000/api/v1/ws/live
```

## Environment Variables

```bash
# .env (backend)
FRED_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
OPENAI_API_KEY=...
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini

# Frontend uses proxy, no env needed
```

## Useful Commands

```bash
# Backend
uvicorn api.main:app --reload --port 8000
pytest tests/  # If you add tests

# Frontend
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build

# Both
# Terminal 1: Backend
cd api && uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```
