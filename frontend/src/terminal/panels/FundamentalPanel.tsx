import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface CompanyAnalysis {
  ticker?: string;
  company_name?: string;
  fundamental_analysis?: {
    profile?: { name?: string };
    financials?: Record<string, unknown>;
    ratios?: Record<string, unknown>;
  };
  valuation?: Record<string, unknown> | { error?: string };
  risk_metrics?: Record<string, unknown> | { error?: string };
  summary?: Record<string, unknown>;
}

function parseCompanyAnalysis(json: unknown): CompanyAnalysis | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as CompanyAnalysis;
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
        <div className="panel-body-muted">Loading…</div>
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
  const ratios = fa?.ratios as Record<string, number> | undefined;
  const valuation = data?.valuation as Record<string, unknown> | undefined;
  const risk = data?.risk_metrics as Record<string, unknown> | undefined;

  return (
    <section className="panel panel-main">
      <div className="panel-title">
        Fundamental: {data?.company_name ?? primarySymbol}
      </div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        {ratios && Object.keys(ratios).length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ color: "var(--accent)", marginBottom: 6 }}>Key ratios</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <tbody>
                {Object.entries(ratios).slice(0, 12).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>{k}</td>
                    <td className="num-mono" style={{ textAlign: "right" }}>
                      {typeof v === "number" ? v.toFixed(2) : String(v)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {valuation && !("error" in valuation) && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ color: "var(--accent)", marginBottom: 6 }}>Valuation (DCF)</div>
            <pre style={{ margin: 0, fontSize: 11, color: "var(--text-soft)" }}>
              {JSON.stringify(valuation, null, 2).slice(0, 400)}…
            </pre>
          </div>
        )}
        {risk && !("error" in risk) && (
          <div>
            <div style={{ color: "var(--accent)", marginBottom: 6 }}>Risk metrics</div>
            <pre style={{ margin: 0, fontSize: 11, color: "var(--text-soft)" }}>
              {JSON.stringify(risk, null, 2).slice(0, 300)}…
            </pre>
          </div>
        )}
        {!ratios && !valuation && !risk && (
          <div className="panel-body-muted">No fundamental data returned.</div>
        )}
      </div>
    </section>
  );
};
