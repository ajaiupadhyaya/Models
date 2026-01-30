import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface Candle {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
}

export const PrimaryInstrument: React.FC = () => {
  const [symbol] = useState("AAPL");
  const [data, setData] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [chartError, setChartError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setChartError(null);
        const res = await fetch(`/api/v1/backtest/sample-data?symbol=${symbol}&period=3mo`);
        const json = await res.json().catch(() => ({}));
        if (!res.ok) {
          setChartError(json?.error ?? `HTTP ${res.status}`);
          setData([]);
          return;
        }
        const candles: Candle[] = (json.candles ?? []).map((c: any) => ({
          date: new Date(c.date),
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close)
        }));
        setData(candles);
      } catch (err) {
        setChartError(err instanceof Error ? err.message : "Failed to load");
        setData([]);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [symbol]);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;

    const width = el.clientWidth || 600;
    const height = 320;

    d3.select(el).selectAll("*").remove();
    if (!data.length) {
      d3.select(el)
        .append("div")
        .attr("class", "panel-empty")
        .text(loading ? "Loading price data…" : (chartError ?? "No data"));
      return;
    }

    const margin = { top: 10, right: 40, bottom: 20, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(data.map(d => d.date.toISOString()))
      .range([0, innerWidth])
      .padding(0.3);

    const y = d3
      .scaleLinear()
      .domain([
        d3.min(data, d => d.low) ?? 0,
        d3.max(data, d => d.high) ?? 1
      ])
      .nice()
      .range([innerHeight, 0]);

    const xAxis = d3
      .axisBottom<Date | string>(d3.scaleTime()
        .domain(d3.extent(data, d => d.date) as [Date, Date])
        .range([0, innerWidth]))
      .ticks(5);

    const yAxis = d3.axisLeft(y).ticks(5);

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("class", "axis axis-x")
      .call(xAxis);

    g.append("g")
      .attr("class", "axis axis-y")
      .call(yAxis);

    const candle = g
      .selectAll("g.candle")
      .data(data)
      .enter()
      .append("g")
      .attr("class", "candle")
      .attr("transform", d => `translate(${x(d.date.toISOString()) ?? 0},0)`);

    // Wicks
    candle
      .append("line")
      .attr("y1", d => y(d.high))
      .attr("y2", d => y(d.low))
      .attr("x1", x.bandwidth() / 2)
      .attr("x2", x.bandwidth() / 2)
      .attr("stroke", "#8b949e");

    // Bodies
    candle
      .append("rect")
      .attr("y", d => y(Math.max(d.open, d.close)))
      .attr("height", d => Math.abs(y(d.open) - y(d.close)) || 1)
      .attr("width", x.bandwidth())
      .attr("fill", d => (d.close >= d.open ? "#3fb950" : "#f85149"));
  }, [data, loading, chartError]);

  return (
    <section className="panel panel-main">
      <div className="panel-title">Primary Instrument: {symbol}</div>
      <div ref={ref} className="chart-root" />
    </section>
  );
};

