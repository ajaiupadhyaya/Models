import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";
import { TimeSeriesLine, Heatmap, YieldCurve } from "../../charts";
import type { TimeSeriesPoint } from "../../charts";

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

/** Macro time-series chart (first series with data) using shared D3 TimeSeriesLine. */
function MacroChart({ series }: { series: MacroSeries[] }) {
  const withData = series.filter((s) => s.data && s.data.length > 1);
  const first = withData[0];
  const data: TimeSeriesPoint[] =
    first?.data?.map((d) => ({ date: new Date(d.date), value: Number(d.value) })).filter((d) => !Number.isNaN(d.value)) ?? [];

  if (!first?.data?.length || data.length < 2) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6, fontSize: 12 }}>Macro trend</div>
      <TimeSeriesLine
        data={data}
        height={180}
        title={first.description || first.series_id}
        className="chart-root"
        style={{ minHeight: 180 }}
      />
    </div>
  );
}

interface CorrelationResponse {
  symbols?: string[];
  matrix?: number[][];
  error?: string;
}

/** Correlation heatmap using shared D3 Heatmap. */
function CorrelationHeatmap({ symbolsParam }: { symbolsParam: string }) {
  const url = `/api/v1/data/correlation?symbols=${encodeURIComponent(symbolsParam)}&period=1y`;
  const { data } = useFetchWithRetry<CorrelationResponse | null>(url, {
    parse: (json) => (json && typeof json === "object" && !("detail" in (json as object)) ? (json as CorrelationResponse) : null),
    deps: [symbolsParam],
  });

  if (!data?.symbols?.length || !data?.matrix?.length || data.error) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6, fontSize: 12 }}>Correlation (returns, 1Y)</div>
      <Heatmap symbols={data.symbols} matrix={data.matrix} className="chart-root" />
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
  const { data } = useFetchWithRetry<YieldCurveResponse | null>("/api/v1/data/yield-curve", {
    parse: (json) => (json && typeof json === "object" && !("detail" in (json as object)) ? (json as YieldCurveResponse) : null),
    deps: [],
  });

  if (!data?.maturities?.length || !data?.yields?.length || data.error) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6, fontSize: 12 }}>Yield curve</div>
      <YieldCurve maturities={data.maturities} yields={data.yields} date={data.date} title="US Treasury yield curve" className="chart-root" style={{ minHeight: 160 }} />
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
