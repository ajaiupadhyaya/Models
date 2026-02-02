import React, { useEffect } from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useWebSocketPrice } from "../../hooks/useWebSocketPrice";
import { useTerminal } from "../TerminalContext";

interface MarketSymbol {
  symbol: string;
  price: number;
  changePct: number;
}

const WATCHLIST = "AAPL,MSFT,GOOGL,TSLA,SPY,QQQ";

interface MarketSummaryResponse {
  detail?: unknown;
  analyses?: Record<string, { price?: number }>;
}

function parseWatchlist(json: unknown): MarketSymbol[] | null {
  const r = json as MarketSummaryResponse;
  if (r?.detail) return null;
  const analyses = (r?.analyses ?? {}) as Record<string, { price?: number }>;
  return Object.entries(analyses).map(([sym, value]) => ({
    symbol: String(sym),
    price: Number(value?.price ?? 0),
    changePct: 0,
  }));
}

export const MarketOverview: React.FC = () => {
  const { primarySymbol, setPrimarySymbol } = useTerminal();
  const url = `/api/v1/ai/market-summary?symbols=${WATCHLIST}`;
  const { data, error, loading, retry } = useFetchWithRetry<MarketSymbol[] | null>(url, {
    parse: parseWatchlist,
    deps: [],
  });
  const { price: wsPrice } = useWebSocketPrice(primarySymbol);

  useEffect(() => {
    const id = setInterval(retry, 60_000);
    return () => clearInterval(id);
  }, [retry]);

  const symbols = (data ?? []).map((p) =>
    p.symbol === primarySymbol && wsPrice != null ? { ...p, price: wsPrice } : p
  );

  return (
    <div className="panel panel-left">
      <div className="panel-title">Watchlist</div>
      {loading && symbols.length === 0 && !error && (
        <div className="panel-body-muted" style={{ fontSize: 12 }}>Loading…</div>
      )}
      {error && (
        <div className="panel-body-muted" style={{ fontSize: 11 }}>
          {error}. Check that the API is running on port 8000.
          <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={retry}>Retry</button>
        </div>
      )}
      {symbols.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {symbols.map((row) => (
            <div
              key={row.symbol}
              role="button"
              tabIndex={0}
              className={`watchlist-row ${row.symbol === primarySymbol ? "selected" : ""}`}
              onClick={() => setPrimarySymbol(row.symbol)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setPrimarySymbol(row.symbol);
                }
              }}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "6px 8px",
                borderRadius: 2,
              }}
            >
              <span className="num-mono" style={{ color: "var(--accent)", fontWeight: 500 }}>
                {row.symbol}
              </span>
              <span className="num-mono" style={{ marginLeft: 8 }}>{row.price.toFixed(2)}</span>
              <span className={`num-mono ${row.changePct >= 0 ? "up" : "down"}`} style={{ marginLeft: 8, minWidth: 48, textAlign: "right" }}>
                {row.changePct !== 0 ? `${row.changePct >= 0 ? "+" : ""}${row.changePct.toFixed(2)}%` : "—"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
