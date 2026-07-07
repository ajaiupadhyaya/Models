import React, { useEffect, useState } from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useWebSocketPrice } from "../../hooks/useWebSocketPrice";
import { useTerminal } from "../TerminalContext";
import { resolveApiUrl } from "../../apiBase";
import { getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { TimeSeriesLine } from "../../charts";
import type { TimeSeriesPoint } from "../../charts";

interface MarketSymbol {
  symbol: string;
  price: number;
  changePct: number;
}

interface MarketSummaryResponse {
  detail?: unknown;
  analyses?: Record<string, { price?: number }>;
}

interface QuotesResponse {
  quotes?: Array<{ symbol: string; price?: number | null; change_pct?: number | null }>;
  source?: string;
  warning?: string;
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

function useSparklineData(symbol: string): TimeSeriesPoint[] {
  const [data, setData] = useState<TimeSeriesPoint[]>([]);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(resolveApiUrl(`/api/v1/backtest/sample-data?symbol=${symbol}&period=1mo`), { headers: getAuthHeaders() });
        const json = await res.json().catch(() => ({}));
        if (!res.ok || cancelled) return;
        const candles = (json.candles ?? []) as Array<{ date: string; close: number }>;
        if (candles.length >= 2) {
          setData(candles.map((c) => ({ date: new Date(c.date), value: c.close })));
        }
      } catch {
        // ignore
      }
    })();
    return () => { cancelled = true; };
  }, [symbol]);
  return data;
}

export const MarketOverview: React.FC = () => {
  const { primarySymbol, setPrimarySymbol, watchlist, setWatchlist } = useTerminal();
  const [addSymbol, setAddSymbol] = useState("");
  const sparklineData = useSparklineData(primarySymbol);
  const symbolsParam = watchlist.length > 0 ? watchlist.join(",") : "AAPL";
  const url = `/api/v1/data/quotes?symbols=${symbolsParam}`;
  const { data: quotesData } = useFetchWithRetry<QuotesResponse | null>(url, {
    parse: (json) => json as QuotesResponse | null,
    deps: [symbolsParam],
  });
  const fallbackUrl = `/api/v1/ai/market-summary?symbols=${symbolsParam}`;
  const { data: summaryData, error, loading, retry } = useFetchWithRetry<MarketSymbol[] | null>(fallbackUrl, {
    parse: parseWatchlist,
    deps: [symbolsParam],
  });
  const hasQuotes = Array.isArray(quotesData?.quotes) && quotesData!.quotes!.some((q) => q.price != null);
  const data = hasQuotes
    ? quotesData!.quotes!.filter((q) => q.price != null).map((q) => ({
        symbol: q.symbol,
        price: Number(q.price ?? 0),
        changePct: Number(q.change_pct ?? 0),
      }))
    : summaryData;
  const fallbackWarning = quotesData?.source === "fallback"
    ? quotesData.warning ?? "Live quotes unavailable; showing fallback demo prices."
    : null;
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
    <div className="flex flex-col h-full bg-surface-container-low text-on-surface">
      <div className="font-label-xs text-label-xs uppercase text-on-tertiary-container tracking-[0.4em] mb-4">WATCHLIST</div>
      {fallbackWarning && (
        <div className="text-tertiary font-data-mono text-[10px] mb-2 uppercase">
          {fallbackWarning}
        </div>
      )}
      <div className="flex gap-2 mb-4 w-full">
        <input
          type="text"
          placeholder="ADD SYMBOL"
          value={addSymbol}
          onChange={(e) => setAddSymbol(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === "Enter" && handleAddSymbol()}
          className="flex-1 bg-background border border-outline-variant text-on-surface px-2 py-1 font-data-mono text-[12px] uppercase focus:outline-none focus:border-on-tertiary-container transition-all"
          aria-label="Add symbol to watchlist"
        />
        <button 
          type="button" 
          className="text-label-xs font-label-xs uppercase tracking-widest px-3 py-1 border border-outline-variant hover:bg-background transition-colors text-on-surface disabled:opacity-50 disabled:cursor-not-allowed" 
          onClick={handleAddSymbol} 
          disabled={!addSymbol.trim()}
        >
          ADD
        </button>
      </div>
      {loading && symbols.length === 0 && !error && (
        <div className="flex flex-col gap-2">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className={`h-3 bg-outline-variant opacity-60 rounded-sm animate-pulse ${i % 3 === 0 ? "w-2/5" : i % 3 === 1 ? "w-3/4" : "w-full"}`}
            />
          ))}
        </div>
      )}
      {error && (
        <div className="text-error font-data-mono text-xs flex items-center gap-2">
          <span>{error}</span>
          <button type="button" className="border border-error px-2 py-1" onClick={retry}>RETRY</button>
        </div>
      )}
      {sparklineData.length >= 2 && (
        <div className="mb-4">
          <div className="text-on-surface-variant font-data-mono text-[10px] mb-1">{primarySymbol} (1M)</div>
          <TimeSeriesLine
            data={sparklineData}
            height={44}
            marginPreset="sparkline"
            showAxis={false}
            strokeWidth={1.2}
            className="w-full"
            style={{ minHeight: 44 }}
          />
        </div>
      )}
      {symbols.length > 0 && (
        <div className="flex flex-col gap-1 overflow-y-auto">
          {symbols.map((row) => (
            <div
              key={row.symbol}
              role="button"
              tabIndex={0}
              className={`flex justify-between items-center px-2 py-2 border-l-2 cursor-pointer transition-colors ${row.symbol === primarySymbol ? "border-primary bg-background" : "border-transparent hover:bg-background/50"}`}
              onClick={() => setPrimarySymbol(row.symbol)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setPrimarySymbol(row.symbol);
                }
              }}
            >
              <span className="font-data-mono text-primary font-medium text-[12px]">
                {row.symbol}
              </span>
              <span className="font-data-mono text-[12px] ml-2">{row.price.toFixed(2)}</span>
              <span className={`font-data-mono text-[12px] ml-2 min-w-[48px] text-right ${row.changePct >= 0 ? "text-accent-green" : "text-error"}`}>
                {row.changePct !== 0 ? `${row.changePct >= 0 ? "+" : ""}${row.changePct.toFixed(2)}%` : "—"}
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveSymbol(row.symbol);
                }}
                className="ml-2 text-on-surface-variant hover:text-on-surface transition-colors"
                title="Remove from watchlist"
                aria-label={`Remove ${row.symbol}`}
              >
                <span className="material-symbols-outlined text-[14px]">close</span>
              </button>
            </div>
          ))}
        </div>
      )}
      {watchlist.length === 0 && !loading && (
        <div className="text-on-surface-variant font-data-mono text-[10px] uppercase mt-2">
          Add symbols above. Watchlist is saved in this browser.
        </div>
      )}
    </div>
  );
};
