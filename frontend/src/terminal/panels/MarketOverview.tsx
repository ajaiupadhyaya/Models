import React, { useEffect, useState } from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useWebSocketPrice } from "../../hooks/useWebSocketPrice";
import { useTerminal } from "../TerminalContext";

interface MarketSymbol {
  symbol: string;
  price: number;
  changePct: number;
}

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
  const { primarySymbol, setPrimarySymbol, watchlist, setWatchlist } = useTerminal();
  const [addSymbol, setAddSymbol] = useState("");
  const symbolsParam = watchlist.length > 0 ? watchlist.join(",") : "AAPL";
  const url = `/api/v1/data/quotes?symbols=${symbolsParam}`;
  const { data: quotesData } = useFetchWithRetry<{ quotes?: Array<{ symbol: string; price?: number | null; change_pct?: number | null }> } | null>(url, {
    parse: (json) => json as { quotes?: Array<{ symbol: string; price?: number | null; change_pct?: number | null }> } | null,
    deps: [symbolsParam],
  });
  const fallbackUrl = `/api/v1/ai/market-summary?symbols=${symbolsParam}`;
  const { data: summaryData, error, loading, retry } = useFetchWithRetry<MarketSymbol[] | null>(fallbackUrl, {
    parse: parseWatchlist,
    deps: [symbolsParam],
  });
  const hasQuotes = Array.isArray(quotesData?.quotes) && quotesData!.quotes!.length > 0;
  const data = hasQuotes
    ? quotesData!.quotes!.map((q) => ({
        symbol: q.symbol,
        price: Number(q.price ?? 0),
        changePct: Number(q.change_pct ?? 0),
      }))
    : summaryData;
  const { price: wsPrice } = useWebSocketPrice(primarySymbol);

  useEffect(() => {
    const id = setInterval(retry, 60_000);
    return () => clearInterval(id);
  }, [retry]);

  const symbols = (data ?? []).map((p) =>
    p.symbol === primarySymbol && wsPrice != null ? { ...p, price: wsPrice } : p
  );

  const handleAddSymbol = () => {
    const sym = addSymbol.trim().toUpperCase();
    if (!sym || watchlist.includes(sym)) return;
    setWatchlist([...watchlist, sym]);
    setAddSymbol("");
  };

  const handleRemoveSymbol = (sym: string) => {
    const next = watchlist.filter((s) => s !== sym);
    setWatchlist(next);
    if (primarySymbol === sym) setPrimarySymbol(next[0] ?? "AAPL");
  };

  return (
    <div className="panel panel-left">
      <div className="panel-title">Watchlist</div>
      <div style={{ marginBottom: 8, display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
        <input
          type="text"
          placeholder="Add symbol"
          value={addSymbol}
          onChange={(e) => setAddSymbol(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === "Enter" && handleAddSymbol()}
          style={{
            flex: 1,
            minWidth: 80,
            background: "var(--bg-panel)",
            border: "1px solid var(--border)",
            color: "var(--text)",
            padding: "4px 8px",
            borderRadius: 4,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
          }}
          aria-label="Add symbol to watchlist"
        />
        <button type="button" className="ai-button" onClick={handleAddSymbol} disabled={!addSymbol.trim()}>
          Add
        </button>
      </div>
      {loading && symbols.length === 0 && !error && (
        <div className="panel-skeleton">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className={`panel-skeleton-line ${i % 3 === 0 ? "short" : i % 3 === 1 ? "medium" : ""}`}
            />
          ))}
        </div>
      )}
      {error && (
        <div className="panel-error-inline">
          <span>{error}</span>
          <button type="button" className="ai-button" onClick={retry}>Retry</button>
        </div>
      )}
      {symbols.length > 0 && (
        <div className="watchlist-rows">
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
                padding: "var(--space-2) var(--space-3)",
                borderRadius: "var(--radius-sm)",
              }}
            >
              <span className="num-mono" style={{ color: "var(--accent)", fontWeight: 500 }}>
                {row.symbol}
              </span>
              <span className="num-mono" style={{ marginLeft: 8 }}>{row.price.toFixed(2)}</span>
              <span className={`num-mono ${row.changePct >= 0 ? "up" : "down"}`} style={{ marginLeft: 8, minWidth: 48, textAlign: "right" }}>
                {row.changePct !== 0 ? `${row.changePct >= 0 ? "+" : ""}${row.changePct.toFixed(2)}%` : "—"}
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveSymbol(row.symbol);
                }}
                style={{
                  marginLeft: 4,
                  background: "none",
                  border: "none",
                  color: "var(--text-soft)",
                  cursor: "pointer",
                  padding: 2,
                  fontSize: 14,
                  lineHeight: 1,
                }}
                title="Remove from watchlist"
                aria-label={`Remove ${row.symbol}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      {watchlist.length === 0 && !loading && (
        <div className="panel-body-muted" style={{ fontSize: 12 }}>Add symbols above. Watchlist is saved in this browser.</div>
      )}
    </div>
  );
};
