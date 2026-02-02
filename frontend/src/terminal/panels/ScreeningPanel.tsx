import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface ScreenRow {
  symbol: string;
  [key: string]: unknown;
}

interface SectorsResponse {
  detail?: unknown;
  sectors?: string[];
  results?: ScreenRow[];
}

function parseSectors(json: unknown): ScreenRow[] | null {
  const r = json as SectorsResponse & unknown[];
  if (r && typeof r === "object" && "detail" in r && (r as SectorsResponse).detail) return null;
  const obj = r as SectorsResponse;
  if (Array.isArray(obj?.sectors)) {
    return (obj.sectors as string[]).map((s) => ({ symbol: s }));
  }
  if (Array.isArray(obj)) {
    return (obj as string[]).map((s) => ({ symbol: s }));
  }
  if (Array.isArray(obj?.results)) return obj.results ?? null;
  return [];
}

export const ScreeningPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const { data, error, loading, retry } = useFetchWithRetry<ScreenRow[] | null>("/api/v1/company/sectors", {
    parse: parseSectors,
    deps: [primarySymbol],
  });

  const rows = data ?? [];

  if (loading) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Screening & discovery</div>
        <div className="panel-body-muted">Loading…</div>
      </section>
    );
  }

  if (error) {
    return (
      <PanelErrorState
        title="Screening & discovery"
        error={error}
        hint="Ensure API is running."
        onRetry={retry}
      />
    );
  }

  return (
    <section className="panel panel-main">
      <div className="panel-title">Screening & discovery</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        {rows.length > 0 ? (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Symbol / Sector</th>
                {Object.keys(rows[0] ?? {}).filter((k) => k !== "symbol").slice(0, 4).map((k) => (
                  <th key={k} style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 20).map((row, i) => (
                <tr key={i}>
                  <td style={{ color: "var(--accent)" }}>{row.symbol}</td>
                  {Object.entries(row)
                    .filter(([k]) => k !== "symbol")
                    .slice(0, 4)
                    .map(([k, v]) => (
                      <td key={k} className="num-mono" style={{ textAlign: "right" }}>
                        {typeof v === "number" ? (v as number).toFixed(2) : String(v)}
                      </td>
                    ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="panel-body-muted">
            No screener results. Multi-factor screener API can be wired here.
          </div>
        )}
      </div>
    </section>
  );
};
