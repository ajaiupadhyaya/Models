import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
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

function parseMacro(json: unknown): MacroSeries[] | null {
  const r = json as MacroResponse;
  if (r?.detail) return null;
  if (Array.isArray(r?.series)) return r.series as MacroSeries[];
  if (r?.series && !Array.isArray(r.series)) return [r.series as MacroSeries];
  return null;
}

export const EconomicPanel: React.FC = () => {
  const { data, error, loading, retry } = useFetchWithRetry<MacroSeries[] | null>("/api/v1/data/macro", {
    parse: parseMacro,
  });

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
          <div className="panel-body-muted">No macro series returned.</div>
        )}
      </div>
    </section>
  );
};
