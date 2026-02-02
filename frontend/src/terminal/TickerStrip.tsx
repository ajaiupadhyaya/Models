import React, { useEffect, useState } from "react";
import { resolveApiUrl } from "../apiBase";
import { useWebSocketPrice } from "../hooks/useWebSocketPrice";
import { getAuthHeaders } from "../hooks/useFetchWithRetry";
import { useTerminal } from "./TerminalContext";

interface TickerItem {
  symbol: string;
  price: number;
  changePct: number;
}

interface TickerStripProps {
  primarySymbol: string;
  onWsStatus?: (connected: boolean) => void;
}

export const TickerStrip: React.FC<TickerStripProps> = ({ primarySymbol, onWsStatus }) => {
  const { watchlist } = useTerminal();
  const [items, setItems] = useState<TickerItem[]>([]);
  const { price: wsPrice, connected: wsConnected } = useWebSocketPrice(primarySymbol);

  useEffect(() => {
    onWsStatus?.(wsConnected);
  }, [wsConnected, onWsStatus]);

  useEffect(() => {
    if (watchlist.length === 0) {
      setItems([]);
      return;
    }
    const fetchData = async () => {
      try {
        const quotesRes = await fetch(resolveApiUrl(`/api/v1/data/quotes?symbols=${watchlist.join(",")}`), { headers: getAuthHeaders() });
        const quotesJson = await quotesRes.json().catch(() => ({}));
        const quoteList = (quotesJson.quotes ?? []) as Array<{ symbol: string; price?: number | null; change_pct?: number | null }>;
        if (quoteList.length > 0) {
          const parsed: TickerItem[] = quoteList.map((q) => ({
            symbol: q.symbol,
            price: Number(q.price ?? 0),
            changePct: Number(q.change_pct ?? 0),
          }));
          setItems(parsed);
          return;
        }
        const res = await fetch(resolveApiUrl(`/api/v1/ai/market-summary?symbols=${watchlist.join(",")}`), { headers: getAuthHeaders() });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) return;
        const parsed: TickerItem[] = Object.entries((json.analyses ?? {}) as Record<string, { price?: number }>).map(
          ([sym, value]) => ({
            symbol: String(sym),
            price: Number(value?.price ?? 0),
            changePct: 0,
          })
        );
        setItems(parsed);
      } catch {
        setItems(watchlist.map((s) => ({ symbol: s, price: 0, changePct: 0 })));
      }
    };
    fetchData();
    const id = setInterval(fetchData, 30_000);
    return () => clearInterval(id);
  }, [watchlist]);

  useEffect(() => {
    if (wsPrice == null || !items.length) return;
    setItems((prev) =>
      prev.map((p) => (p.symbol === primarySymbol ? { ...p, price: wsPrice } : p))
    );
  }, [wsPrice, primarySymbol, items.length]);

  if (items.length === 0) {
    return (
      <div className="ticker-strip">
        <span className="num-mono" style={{ color: "var(--text-soft)" }}>
          Loading tickers…
        </span>
      </div>
    );
  }

  return (
    <div className="ticker-strip">
      {items.map((item) => (
        <span
          key={item.symbol}
          style={{
            marginRight: 24,
            whiteSpace: "nowrap",
          }}
        >
          <span className="num-mono" style={{ marginRight: 6, color: "var(--accent)" }}>
            {item.symbol}
          </span>
          <span className="num-mono" style={{ marginRight: 4 }}>
            {item.price > 0 ? item.price.toFixed(2) : "—"}
          </span>
          <span className={`num-mono ${item.changePct >= 0 ? "up" : "down"}`}>
            {item.changePct !== 0 ? `${item.changePct >= 0 ? "+" : ""}${item.changePct.toFixed(2)}%` : ""}
          </span>
        </span>
      ))}
    </div>
  );
};
