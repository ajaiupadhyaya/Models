import React, { useEffect, useState } from "react";
import { useWebSocketPrice } from "../hooks/useWebSocketPrice";

const WATCHLIST = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"];

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
  const [items, setItems] = useState<TickerItem[]>([]);
  const { price: wsPrice, connected: wsConnected } = useWebSocketPrice(primarySymbol);

  useEffect(() => {
    onWsStatus?.(wsConnected);
  }, [wsConnected, onWsStatus]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`/api/v1/ai/market-summary?symbols=${WATCHLIST.join(",")}`);
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
        setItems(WATCHLIST.map((s) => ({ symbol: s, price: 0, changePct: 0 })));
      }
    };
    fetchData();
    const id = setInterval(fetchData, 30_000);
    return () => clearInterval(id);
  }, []);

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
