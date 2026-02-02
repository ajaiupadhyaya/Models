import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface CompanyAnalysis {
  ticker?: string;
  company_name?: string;
  fundamental_analysis?: {
    profile?: { name?: string };
    valuation?: Record<string, unknown>;
    profitability?: Record<string, unknown>;
    financial_health?: Record<string, unknown>;
    growth?: Record<string, unknown>;
    financials?: Record<string, unknown>;
    ratios?: Record<string, unknown>;
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
        hint="Ensure API is running on port 8000."
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
