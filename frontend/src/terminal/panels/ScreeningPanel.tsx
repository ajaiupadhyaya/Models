import React, { useState, useCallback } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface ScreenRow {
  symbol: string;
  name?: string;
  sector?: string;
  market_cap?: number;
  industry?: string;
}

interface SectorsResponse {
  detail?: unknown;
  sectors?: string[];
}

interface ScreenerResponse {
  detail?: unknown;
  results?: ScreenRow[];
  count?: number;
  error?: string;
}

function parseSectors(json: unknown): string[] | null {
  const r = json as SectorsResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.sectors) ? r.sectors : [];
}

function parseScreener(json: unknown): ScreenRow[] | null {
  const r = json as ScreenerResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.results) ? r.results : [];
}

export const ScreeningPanel: React.FC = () => {
  const { primarySymbol, setPrimarySymbol } = useTerminal();
  const [selectedSector, setSelectedSector] = useState<string>("");
  const [minMarketCap, setMinMarketCap] = useState<string>("");
  const [screenerResults, setScreenerResults] = useState<ScreenRow[] | null>(null);
  const [screenerLoading, setScreenerLoading] = useState(false);
  const [screenerError, setScreenerError] = useState<string | null>(null);

  const { data: sectorsList, error: sectorsError, loading: sectorsLoading, retry: sectorsRetry } = useFetchWithRetry<string[] | null>(
    "/api/v1/company/sectors",
    { parse: parseSectors }
  );
  const sectors = sectorsList ?? [];

  const runScreen = useCallback(async () => {
    setScreenerLoading(true);
    setScreenerError(null);
    try {
      const params = new URLSearchParams();
      if (selectedSector) params.set("sector", selectedSector);
      const minCap = minMarketCap.trim() ? parseFloat(minMarketCap) : undefined;
      if (minCap != null && !Number.isNaN(minCap)) params.set("min_market_cap", String(minCap));
      params.set("limit", "30");
      const res = await fetch(resolveApiUrl(`/api/v1/screener/run?${params.toString()}`), { headers: getAuthHeaders() });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setScreenerError((json?.detail ?? json?.error ?? `HTTP ${res.status}`) as string);
        setScreenerResults(null);
        return;
      }
      const results = Array.isArray((json as ScreenerResponse).results) ? (json as ScreenerResponse).results : [];
      setScreenerResults(results);
    } catch (err) {
      setScreenerError(err instanceof Error ? err.message : "Request failed");
      setScreenerResults(null);
    } finally {
      setScreenerLoading(false);
    }
  }, [selectedSector, minMarketCap]);

  if (sectorsLoading && sectors.length === 0) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Screening & discovery</div>
        <div className="panel-body-muted">Loading…</div>
      </section>
    );
  }

  if (sectorsError && sectors.length === 0) {
    return (
      <PanelErrorState
        title="Screening & discovery"
        error={sectorsError}
        hint="Ensure API is running."
        onRetry={sectorsRetry}
      />
    );
  }

  const rows = screenerResults ?? [];

  return (
    <section className="panel panel-main">
      <div className="panel-title">Screening & discovery</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12, display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
          <select
            value={selectedSector}
            onChange={(e) => setSelectedSector(e.target.value)}
            style={{
              background: "var(--bg-panel)",
              border: "1px solid var(--border)",
              color: "var(--text)",
              padding: "6px 8px",
              borderRadius: 4,
              fontSize: 12,
              minWidth: 140,
            }}
            aria-label="Sector"
          >
            <option value="">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <input
            type="number"
            className="ai-input"
            placeholder="Min market cap (optional)"
            value={minMarketCap}
            onChange={(e) => setMinMarketCap(e.target.value)}
            style={{ width: 140 }}
            aria-label="Min market cap"
          />
          <button
            type="button"
            className="ai-button"
            disabled={screenerLoading}
            onClick={runScreen}
          >
            {screenerLoading ? "Running…" : "Run screen"}
          </button>
        </div>
        {screenerError && (
          <div style={{ color: "var(--accent-red)", marginBottom: 8, fontSize: 11 }}>{screenerError}</div>
        )}
        {rows.length > 0 ? (
          <>
            <div style={{ marginBottom: 8, display: "flex", justifyContent: "flex-end" }}>
              <button
                type="button"
                className="ai-button"
                onClick={() => {
                  const headers = ["Symbol", "Name", "Sector", "Market Cap"];
                  const csvRows = [headers.join(","), ...rows.map((r) => [
                    `"${(r.symbol ?? "").replace(/"/g, '""')}"`,
                    `"${(r.name ?? "").replace(/"/g, '""')}"`,
                    `"${(r.sector ?? "").replace(/"/g, '""')}"`,
                    r.market_cap != null ? String(r.market_cap) : "",
                  ].join(","))];
                  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `screener-${new Date().toISOString().slice(0, 10)}.csv`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                Export CSV
              </button>
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Symbol</th>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Name</th>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Sector</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Market cap</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                <tr key={i}>
                  <td>
                    <button
                      type="button"
                      onClick={() => setPrimarySymbol(row.symbol)}
                      style={{
                        background: "none",
                        border: "none",
                        color: "var(--accent)",
                        cursor: "pointer",
                        padding: 0,
                        font: "inherit",
                        textDecoration: row.symbol === primarySymbol ? "underline" : "none",
                      }}
                    >
                      {row.symbol}
                    </button>
                  </td>
                  <td style={{ color: "var(--text)", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }} title={row.name}>
                    {row.name ?? "—"}
                  </td>
                  <td style={{ color: "var(--text-soft)" }}>{row.sector ?? "—"}</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>
                    {row.market_cap != null ? `${(row.market_cap / 1e9).toFixed(2)}B` : "—"}
                  </td>
                </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <div className="panel-body-muted">
            Select sector (optional), set min market cap (optional), then Run screen.
          </div>
        )}
      </div>
    </section>
  );
};
