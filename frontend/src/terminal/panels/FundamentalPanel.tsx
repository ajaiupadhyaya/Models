import React, { useEffect, useState } from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { resolveApiUrl } from "../../apiBase";
import { getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { PanelErrorState } from "./PanelErrorState";
import { TimeSeriesLine } from "../../charts";
import type { TimeSeriesPoint } from "../../charts";

interface CompanyAnalysis {
  ticker?: string;
  company_name?: string;
  fundamental_analysis?: {
    profile?: { name?: string; sector?: string };
    valuation?: Record<string, unknown>;
    profitability?: Record<string, unknown>;
    financial_health?: Record<string, unknown>;
    growth?: Record<string, unknown>;
    financials?: Record<string, unknown>;
    ratios?: Record<string, unknown>;
    financials_summary?: {
      income?: Record<string, number | string>;
      balance_sheet?: Record<string, number | string>;
      cash_flow?: Record<string, number | string>;
    };
  };
  valuation?: Record<string, unknown> | { error?: string };
  risk_metrics?: Record<string, unknown> | { error?: string };
  summary?: {
    overall_grade?: string;
    overall_score?: number;
    recommendation?: string;
    valuation_grade?: { grade?: string };
    profitability_grade?: { grade?: string };
    financial_health_grade?: { grade?: string };
    growth_grade?: { grade?: string };
  };
}

function parseCompanyAnalysis(json: unknown): CompanyAnalysis | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as CompanyAnalysis;
}

interface SectorCompany {
  symbol?: string;
  ticker?: string;
  name?: string;
  sector?: string;
  market_cap?: number;
}

interface SectorResponse {
  sector?: string;
  companies?: SectorCompany[];
  detail?: unknown;
}

function parseSector(json: unknown): SectorCompany[] | null {
  const r = json as SectorResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.companies) ? r.companies : [];
}

type PeerSortKey = "symbol" | "name" | "market_cap";
type PeerSortDir = "asc" | "desc";

/** Peer comparison table (sector peers) – sortable columns and CSV export. */
function PeerComparisonTable({ sector, primarySymbol }: { sector: string; primarySymbol: string }) {
  const [sortKey, setSortKey] = useState<PeerSortKey>("symbol");
  const [sortDir, setSortDir] = useState<PeerSortDir>("asc");
  const url = sector ? `/api/v1/company/sector/${encodeURIComponent(sector)}?limit=12` : null;
  const { data: peers, loading } = useFetchWithRetry<SectorCompany[] | null>(url, {
    parse: parseSector,
    deps: [sector],
  });
  const list = peers ?? [];
  const sym = (c: SectorCompany) => c.symbol ?? c.ticker ?? "";
  const filtered = list.filter((c) => sym(c) !== primarySymbol).slice(0, 10);
  const rows = [...filtered].sort((a, b) => {
    let va: string | number = sortKey === "symbol" ? sym(a) : sortKey === "name" ? (a.name ?? "") : (a.market_cap ?? 0);
    let vb: string | number = sortKey === "symbol" ? sym(b) : sortKey === "name" ? (b.name ?? "") : (b.market_cap ?? 0);
    if (sortKey === "market_cap") {
      return sortDir === "asc" ? (va as number) - (vb as number) : (vb as number) - (va as number);
    }
    const cmp = String(va).toLowerCase().localeCompare(String(vb).toLowerCase());
    return sortDir === "asc" ? cmp : -cmp;
  });
  const toggleSort = (key: PeerSortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };
  if (!sector || sector === "N/A") return null;
  if (loading) return <div className="panel-body-muted" style={{ fontSize: 11 }}>Loading peers…</div>;
  if (rows.length === 0) return <div className="panel-body-muted" style={{ fontSize: 11 }}>No sector peers found.</div>;
  const exportCsv = () => {
    const headers = ["Symbol", "Name", "Sector", "Market Cap"];
    const csvRows = [headers.join(","), ...rows.map((c) => [
      `"${(sym(c) || "").replace(/"/g, '""')}"`,
      `"${(c.name ?? "").replace(/"/g, '""')}"`,
      `"${(c.sector ?? "").replace(/"/g, '""')}"`,
      c.market_cap != null ? String(c.market_cap) : "",
    ].join(","))];
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = `peers-${sector.replace(/\s+/g, "-")}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(downloadUrl);
  };
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ color: "var(--accent)", fontSize: 11 }}>Sector peers ({sector})</span>
        <button type="button" className="ai-button" style={{ padding: "2px 6px", fontSize: 10 }} onClick={exportCsv}>Export CSV</button>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead>
          <tr>
            {(["symbol", "name", "market_cap"] as PeerSortKey[]).map((key) => (
              <th
                key={key}
                style={{
                  textAlign: key === "market_cap" ? "right" : "left",
                  color: "var(--text-soft)",
                  fontWeight: 500,
                  cursor: "pointer",
                  userSelect: "none",
                }}
                onClick={() => toggleSort(key)}
              >
                {key === "market_cap" ? "Market cap" : key.charAt(0).toUpperCase() + key.slice(1)}
                {sortKey === key && (sortDir === "asc" ? " ↑" : " ↓")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={sym(c)}>
              <td className="num-mono" style={{ color: "var(--accent)", padding: "2px 8px 2px 0" }}>{sym(c) || "—"}</td>
              <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }}>{c.name ?? "—"}</td>
              <td className="num-mono" style={{ textAlign: "right" }}>{c.market_cap != null ? `${(c.market_cap / 1e9).toFixed(2)}B` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Price trend chart for the primary symbol (1y) using shared D3 TimeSeriesLine. */
function PriceTrendChart({ symbol }: { symbol: string }) {
  const [candles, setCandles] = useState<Array<{ date: string; close: number }>>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(resolveApiUrl(`/api/v1/backtest/sample-data?symbol=${symbol}&period=1y`), { headers: getAuthHeaders() });
        const json = await res.json().catch(() => ({}));
        if (!res.ok || cancelled) return;
        const list = (json.candles ?? []) as Array<{ date: string; close: number }>;
        if (list.length) setCandles(list);
      } catch {
        // ignore
      }
    })();
    return () => { cancelled = true; };
  }, [symbol]);

  const data: TimeSeriesPoint[] = candles
    .map((d) => ({ date: new Date(d.date), value: Number(d.close) }))
    .filter((d) => !Number.isNaN(d.value));

  if (data.length < 2) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ color: "var(--accent)", marginBottom: 4, fontSize: 11 }}>Price trend (1Y)</div>
      <TimeSeriesLine data={data} height={160} marginPreset="compact" className="chart-root" style={{ minHeight: 160 }} />
    </div>
  );
}

function renderKeyValueTable(
  title: string,
  obj: Record<string, unknown> | undefined,
  maxRows: number = 10,
  formatVal: (v: unknown) => string = (v) => (typeof v === "number" ? (v as number).toFixed(2) : String(v ?? "—"))
) {
  if (!obj || Object.keys(obj).length === 0) return null;
  const entries = Object.entries(obj).filter(([, v]) => v != null && v !== "").slice(0, maxRows);
  if (entries.length === 0) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ color: "var(--accent)", marginBottom: 6 }}>{title}</div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>{k.replace(/_/g, " ")}</td>
              <td className="num-mono" style={{ textAlign: "right" }}>{formatVal(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const FundamentalPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const url = `/api/v1/company/analyze/${primarySymbol}?include_dcf=true&include_risk=true&include_technicals=false`;
  const { data, error, loading, retry } = useFetchWithRetry<CompanyAnalysis | null>(url, {
    parse: parseCompanyAnalysis,
    deps: [primarySymbol],
  });

  if (loading) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Fundamental: {primarySymbol}</div>
        <div className="panel-skeleton">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div
              key={i}
              className={`panel-skeleton-line ${i % 3 === 0 ? "short" : i % 3 === 1 ? "medium" : ""}`}
            />
          ))}
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <PanelErrorState
        title={`Fundamental: ${primarySymbol}`}
        error={error}
        hint="Try again or ensure the service is reachable."
        onRetry={retry}
      />
    );
  }

  const fa = data?.fundamental_analysis;
  const valuation = data?.valuation as Record<string, unknown> | undefined;
  const risk = data?.risk_metrics as Record<string, unknown> | undefined;
  const summary = data?.summary;
  const ratios = fa?.ratios as Record<string, number> | undefined;
  const faValuation = fa?.valuation as Record<string, unknown> | undefined;
  const profitability = fa?.profitability as Record<string, unknown> | undefined;
  const financialHealth = fa?.financial_health as Record<string, unknown> | undefined;
  const growth = fa?.growth as Record<string, unknown> | undefined;

  const hasContent =
    (ratios && Object.keys(ratios).length > 0) ||
    faValuation ||
    profitability ||
    financialHealth ||
    growth ||
    summary?.overall_grade ||
    (valuation && !("error" in valuation)) ||
    (risk && !("error" in risk));

  return (
    <section className="panel panel-main">
      <div className="panel-title">
        Fundamental: {data?.company_name ?? primarySymbol}
      </div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <PriceTrendChart symbol={primarySymbol} />
        {fa?.profile?.sector && <PeerComparisonTable sector={fa.profile.sector} primarySymbol={primarySymbol} />}
        {fa?.financials_summary && (fa.financials_summary.income || fa.financials_summary.balance_sheet || fa.financials_summary.cash_flow) && (
          <>
            {renderKeyValueTable("Income (latest)", fa.financials_summary.income, 8, (v) => typeof v === "number" && Math.abs(v) >= 1e6 ? `${((v as number) / 1e6).toFixed(1)}M` : typeof v === "number" ? (v as number).toFixed(2) : String(v ?? "—"))}
            {renderKeyValueTable("Balance sheet (latest)", fa.financials_summary.balance_sheet, 8, (v) => typeof v === "number" && Math.abs(v) >= 1e6 ? `${((v as number) / 1e6).toFixed(1)}M` : typeof v === "number" ? (v as number).toFixed(2) : String(v ?? "—"))}
            {renderKeyValueTable("Cash flow (latest)", fa.financials_summary.cash_flow, 8, (v) => typeof v === "number" && Math.abs(v) >= 1e6 ? `${((v as number) / 1e6).toFixed(1)}M` : typeof v === "number" ? (v as number).toFixed(2) : String(v ?? "—"))}
          </>
        )}
        {summary && (summary.overall_grade || summary.recommendation) && (
          <div style={{ marginBottom: 12, padding: 8, background: "var(--bg-panel)", borderRadius: 4, border: "1px solid var(--border)" }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>AI company health</div>
            {summary.overall_grade && (
              <span className="num-mono" style={{ marginRight: 12 }}>Grade: {summary.overall_grade}</span>
            )}
            {summary.overall_score != null && (
              <span className="num-mono" style={{ marginRight: 12 }}>Score: {Number(summary.overall_score).toFixed(1)}</span>
            )}
            {summary.recommendation && (
              <div style={{ color: "var(--text-soft)", fontSize: 11, marginTop: 4 }}>{summary.recommendation}</div>
            )}
          </div>
        )}
        {renderKeyValueTable("Valuation (P/E, P/B, EV)", faValuation ?? ratios ?? {}, 10, (v) =>
          typeof v === "number" && (v as number) > 1000 ? (v as number).toLocaleString() : (typeof v === "number" ? (v as number).toFixed(2) : String(v ?? "—"))
        )}
        {renderKeyValueTable("Profitability (margins, ROE, ROA)", profitability)}
        {renderKeyValueTable("Financial health (leverage, liquidity)", financialHealth)}
        {renderKeyValueTable("Growth", growth)}
        {valuation && !("error" in valuation) && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ color: "var(--accent)", marginBottom: 6 }}>DCF valuation</div>
            <pre style={{ margin: 0, fontSize: 11, color: "var(--text-soft)" }}>
              {JSON.stringify(valuation, null, 2).slice(0, 500)}{Object.keys(valuation).length > 3 ? "…" : ""}
            </pre>
          </div>
        )}
        {risk && !("error" in risk) && (
          <div>
            <div style={{ color: "var(--accent)", marginBottom: 6 }}>Risk metrics</div>
            <pre style={{ margin: 0, fontSize: 11, color: "var(--text-soft)" }}>
              {JSON.stringify(risk, null, 2).slice(0, 400)}{Object.keys(risk).length > 2 ? "…" : ""}
            </pre>
          </div>
        )}
        {!hasContent && (
          <div className="panel-body-muted">No fundamental data returned. Ensure API and data sources are configured.</div>
        )}
      </div>
    </section>
  );
};
