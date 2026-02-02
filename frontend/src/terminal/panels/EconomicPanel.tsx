import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface MacroPoint {
  date: string;
  value: number;
}

interface MacroSeries {
  series_id: string;
  description: string;
  data: MacroPoint[];
}

interface MacroResponse {
  series?: MacroSeries[] | MacroSeries;
  detail?: unknown;
  error?: string;
}

interface CalendarEvent {
  date?: string;
  release_name?: string;
  release_id?: number;
}

interface CalendarResponse {
  detail?: unknown;
  events?: CalendarEvent[];
  error?: string;
}

function parseMacro(json: unknown): MacroSeries[] | null {
  const r = json as MacroResponse;
  if (r?.detail) return null;
  if (Array.isArray(r?.series)) return r.series as MacroSeries[];
  if (r?.series && !Array.isArray(r.series)) return [r.series as MacroSeries];
  return null;
}

function parseCalendar(json: unknown): CalendarEvent[] | null {
  const r = json as CalendarResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.events) ? r.events : [];
}

/** D3 time-series chart for macro series (first series with data). */
function MacroChart({ series }: { series: MacroSeries[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const withData = series.filter((s) => s.data && s.data.length > 1);
  const first = withData[0];

  useEffect(() => {
    if (!ref.current || !first?.data?.length) return;
    const el = ref.current;
    const width = el.clientWidth || 400;
    const height = 180;
    const margin = { top: 12, right: 12, bottom: 24, left: 44 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(el).selectAll("*").remove();
    const data = first.data.map((d) => ({ date: new Date(d.date), value: Number(d.value) })).filter((d) => !Number.isNaN(d.value));
    if (data.length < 2) return;

    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleTime().domain(d3.extent(data, (d) => d.date) as [Date, Date]).range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(d3.extent(data, (d) => d.value) as [number, number]).nice().range([innerHeight, 0]);

    const line = d3.line<{ date: Date; value: number }>().x((d) => xScale(d.date)).y((d) => yScale(d.value));
    g.append("path").datum(data).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 1.5).attr("d", line);
    g.append("g").attr("transform", `translate(0,${innerHeight})`).attr("class", "axis axis-x").call(d3.axisBottom(xScale).ticks(5));
    g.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yScale).ticks(5));
    g.append("text").attr("x", 2).attr("y", 8).attr("fill", "var(--text-soft)").attr("font-size", 10).attr("font-family", "var(--font-mono)").text(first.description || first.series_id);
  }, [first?.series_id, first?.data?.length]);

  if (!first?.data?.length) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6, fontSize: 12 }}>Macro trend</div>
      <div ref={ref} className="chart-root" style={{ minHeight: 180 }} />
    </div>
  );
}

interface CorrelationResponse {
  symbols?: string[];
  matrix?: number[][];
  error?: string;
}

/** D3 correlation heatmap (global market correlation). */
function CorrelationHeatmap({ symbolsParam }: { symbolsParam: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const url = `/api/v1/data/correlation?symbols=${encodeURIComponent(symbolsParam)}&period=1y`;
  const { data } = useFetchWithRetry<CorrelationResponse | null>(url, {
    parse: (json) => (json && typeof json === "object" && !("detail" in (json as object)) ? (json as CorrelationResponse) : null),
    deps: [symbolsParam],
  });

  useEffect(() => {
    if (!ref.current || !data?.symbols?.length || !data?.matrix?.length) return;
    const el = ref.current;
    const symbols = data.symbols;
    const matrix = data.matrix;
    const n = symbols.length;
    const cellSize = 28;
    const margin = { top: 8, left: 64 };
    const width = margin.left + n * cellSize;
    const height = margin.top + n * cellSize;

    d3.select(el).selectAll("*").remove();
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleLinear<string>().domain([-1, 0, 1]).range(["#2166ac", "#333333", "#b2182b"]);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i]?.[j] ?? 0;
        g.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 2)
          .attr("height", cellSize - 2)
          .attr("fill", colorScale(v))
          .attr("stroke", "var(--border)")
          .attr("stroke-width", 0.5);
        g.append("text")
          .attr("x", j * cellSize + (cellSize - 2) / 2)
          .attr("y", i * cellSize + (cellSize - 2) / 2 + 4)
          .attr("text-anchor", "middle")
          .attr("font-size", 9)
          .attr("font-family", "var(--font-mono)")
          .attr("fill", Math.abs(v) > 0.5 ? "#fff" : "var(--text)")
          .text(typeof v === "number" ? v.toFixed(2) : "");
      }
      g.append("text")
        .attr("x", -6)
        .attr("y", i * cellSize + (cellSize - 2) / 2 + 4)
        .attr("text-anchor", "end")
        .attr("font-size", 9)
        .attr("font-family", "var(--font-mono)")
        .attr("fill", "var(--text-soft)")
        .text(symbols[i] ?? "");
    }
    for (let j = 0; j < n; j++) {
      g.append("text")
        .attr("x", j * cellSize + (cellSize - 2) / 2)
        .attr("y", -4)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("font-family", "var(--font-mono)")
        .attr("fill", "var(--text-soft)")
        .text(symbols[j] ?? "");
    }
  }, [data?.symbols, data?.matrix]);

  if (!data?.symbols?.length || !data?.matrix?.length) return null;
  if (data.error) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6, fontSize: 12 }}>Correlation (returns, 1Y)</div>
      <div ref={ref} className="chart-root" />
    </div>
  );
}

interface YieldCurveResponse {
  maturities?: number[];
  yields?: number[];
  date?: string;
  error?: string;
}

function YieldCurveChart() {
  const ref = useRef<HTMLDivElement | null>(null);
  const { data } = useFetchWithRetry<YieldCurveResponse | null>("/api/v1/data/yield-curve", {
    parse: (json) => (json && typeof json === "object" && !("detail" in (json as object)) ? (json as YieldCurveResponse) : null),
    deps: [],
  });

  useEffect(() => {
    if (!ref.current || !data?.maturities?.length || !data?.yields?.length) return;
    const el = ref.current;
    const maturities = data.maturities;
    const yields = data.yields;
    const n = Math.min(maturities.length, yields.length);
    if (n < 2) return;
    const margin = { top: 12, right: 12, bottom: 28, left: 40 };
    const width = Math.min(el.clientWidth || 320, 400);
    const height = 160;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(el).selectAll("*").remove();
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([Math.min(...maturities.slice(0, n)), Math.max(...maturities.slice(0, n))]).range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain([Math.min(...yields.slice(0, n)) - 0.1, Math.max(...yields.slice(0, n)) + 0.1]).nice().range([innerHeight, 0]);

    const lineData = maturities.slice(0, n).map((m, i) => ({ m, y: yields[i]! }));
    const line = d3.line<{ m: number; y: number }>().x((d) => xScale(d.m)).y((d) => yScale(d.y));
    g.append("path").datum(lineData).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2).attr("d", line);
    g.selectAll("circle").data(lineData).enter().append("circle")
      .attr("cx", (d) => xScale(d.m))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 4)
      .attr("fill", "var(--accent)");
    g.append("g").attr("transform", `translate(0,${innerHeight})`).attr("class", "axis axis-x").call(d3.axisBottom(xScale).tickFormat((d) => d + "Y"));
    g.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => d + "%"));
    g.append("text").attr("x", 2).attr("y", 8).attr("fill", "var(--text-soft)").attr("font-size", 10).attr("font-family", "var(--font-mono)").text("US Treasury yield curve" + (data.date ? ` (${data.date})` : ""));
  }, [data?.maturities, data?.yields, data?.date]);

  if (!data?.maturities?.length || !data?.yields?.length || data.error) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6, fontSize: 12 }}>Yield curve</div>
      <div ref={ref} className="chart-root" style={{ minHeight: 160 }} />
    </div>
  );
}

export const EconomicPanel: React.FC = () => {
  const { watchlist } = useTerminal();
  const correlationSymbols = watchlist.length >= 2 ? watchlist.slice(0, 8).join(",") : "AAPL,MSFT,GOOGL,AMZN,TSLA";
  const { data, error, loading, retry } = useFetchWithRetry<MacroSeries[] | null>("/api/v1/data/macro", {
    parse: parseMacro,
  });
  const { data: calendarData, error: calendarError, loading: calendarLoading } = useFetchWithRetry<CalendarEvent[] | null>(
    "/api/v1/data/economic-calendar?days_ahead=30&limit=25",
    { parse: parseCalendar }
  );
  const calendarEvents = calendarData ?? [];

  if (loading) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Economic indicators</div>
        <div className="panel-skeleton">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className={`panel-skeleton-line ${i % 2 === 0 ? "short" : "medium"}`}
            />
          ))}
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <PanelErrorState
        title="Economic indicators"
        error={error}
        hint="Set FRED_API_KEY and ensure /api/v1/data/macro is available."
        onRetry={retry}
      />
    );
  }

  return (
    <section className="panel panel-main">
      <div className="panel-title">Economic indicators</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <CorrelationHeatmap symbolsParam={correlationSymbols} />
        <YieldCurveChart />
        {data && data.length > 0 && <MacroChart series={data} />}
        {data && data.length > 0 ? (
          data.slice(0, 6).map((s) => (
            <div key={s.series_id} style={{ marginBottom: 12 }}>
              <div style={{ color: "var(--accent)", marginBottom: 4 }}>{s.description || s.series_id}</div>
              {s.data && s.data.length > 0 ? (
                <div className="num-mono" style={{ fontSize: 11, color: "var(--text-soft)" }}>
                  Latest: {s.data[s.data.length - 1]?.date} = {Number(s.data[s.data.length - 1]?.value).toFixed(2)}
                </div>
              ) : (
                <span className="panel-body-muted">No data</span>
              )}
            </div>
          ))
        ) : (
          <div className="panel-body-muted">
            No macro data. Set FRED_API_KEY in .env for economic indicators, or use the API docs to configure data sources.
          </div>
        )}

        <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Economic calendar (upcoming)</div>
          {calendarLoading ? (
            <div className="panel-body-muted">Loading…</div>
          ) : calendarError ? (
            <div style={{ fontSize: 11, color: "var(--text-soft)" }}>Calendar unavailable. Set FRED_API_KEY.</div>
          ) : calendarEvents.length > 0 ? (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Date</th>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Event</th>
                </tr>
              </thead>
              <tbody>
                {calendarEvents.slice(0, 15).map((ev, i) => (
                  <tr key={i}>
                    <td className="num-mono" style={{ color: "var(--text)", padding: "2px 8px 2px 0" }}>{ev.date ?? "—"}</td>
                    <td style={{ color: "var(--text-soft)" }}>{ev.release_name ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="panel-body-muted">No upcoming events. Set FRED_API_KEY for calendar.</div>
          )}
        </div>
      </div>
    </section>
  );
};
