import React, { useEffect, useState } from "react";
import * as d3 from "d3";
import { useWebSocketPrice } from "../../hooks/useWebSocketPrice";

interface MarketSymbol {
  symbol: string;
  price: number;
  changePct: number;
}

const WATCHLIST = "AAPL,MSFT,GOOGL,TSLA,SPY,QQQ";

const FIRST_SYMBOL = "AAPL";

export const MarketOverview: React.FC = () => {
  const [symbols, setSymbols] = useState<MarketSymbol[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { price: wsPrice, connected: wsConnected } = useWebSocketPrice(FIRST_SYMBOL);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setError(null);
        const res = await fetch(`/api/v1/ai/market-summary?symbols=${WATCHLIST}`);
        const json = await res.json().catch(() => ({}));
        if (!res.ok) {
          setError(json?.detail ?? `HTTP ${res.status}`);
          setSymbols([]);
          return;
        }
            const parsed: MarketSymbol[] = Object.entries<any>(json.analyses ?? {}).map(
          ([sym, value]) => ({
            symbol: String(sym),
            price: Number((value as any)?.price ?? 0),
            changePct: 0
          })
        );
        setSymbols(parsed);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
        setSymbols([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (wsPrice == null) return;
    setSymbols((prev) =>
      prev.length && prev[0].symbol === FIRST_SYMBOL
        ? [{ ...prev[0], price: wsPrice }, ...prev.slice(1)]
        : prev
    );
  }, [wsPrice]);

  useEffect(() => {
    // Simple D3 horizontal bar representation of prices
    const container = d3.select("#market-overview-bars");
    container.selectAll("*").remove();

    if (!symbols.length) return;

    const width = 260;
    const barHeight = 18;
    const height = barHeight * symbols.length;

    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const x = d3
      .scaleLinear()
      .domain([0, d3.max(symbols, d => d.price) ?? 1])
      .range([0, width - 80]);

    const g = svg
      .append("g")
      .attr("transform", "translate(70,0)");

    const row = g
      .selectAll("g.row")
      .data(symbols)
      .enter()
      .append("g")
      .attr("class", "row")
      .attr("transform", (_d, i) => `translate(0,${i * barHeight})`);

    row
      .append("rect")
      .attr("height", barHeight - 4)
      .attr("width", d => x(d.price))
      .attr("fill", "#58a6ff");

    row
      .append("text")
      .attr("x", -10)
      .attr("y", (barHeight - 4) / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .text(d => d.symbol);

    row
      .append("text")
      .attr("x", d => x(d.price) + 6)
      .attr("y", (barHeight - 4) / 2)
      .attr("dy", "0.32em")
      .attr("fill", "#8b949e")
      .text(d => d.price.toFixed(2));
  }, [symbols]);

  return (
    <div className="panel panel-left">
      <div className="panel-title">
        Watchlist
        {wsConnected && <span style={{ marginLeft: 6, fontSize: 10, color: "var(--text-soft)" }}>Live</span>}
      </div>
      {loading && symbols.length === 0 && !error && (
        <div className="panel-body-muted" style={{ fontSize: 12 }}>Loading…</div>
      )}
      {error && (
        <div className="panel-body-muted" style={{ fontSize: 11, color: "var(--text-soft)" }}>
          {error}
        </div>
      )}
      <div id="market-overview-bars" />
    </div>
  );
};

