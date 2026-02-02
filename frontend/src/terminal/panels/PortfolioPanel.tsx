import React, { useEffect, useState } from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface DashboardData {
  timestamp?: string;
  system?: { total_predictions?: number; total_errors?: number };
  active_models?: number;
  available_models?: string[];
  recent_predictions?: Array<{ model_name?: string; symbol?: string; signal?: number; confidence?: number }>;
  recent_errors?: Array<{ message?: string }>;
}

interface QuickPredict {
  symbol?: string;
  signal?: number;
  recommendation?: string;
  current_price?: number;
  error?: string;
}

interface RiskMetrics {
  ticker?: string;
  var_95_pct?: number;
  var_99_pct?: number;
  cvar_95_pct?: number;
  cvar_99_pct?: number;
  volatility_annual_pct?: number;
  max_drawdown_pct?: number;
  sharpe_ratio?: number;
}

function parseDashboard(json: unknown): DashboardData | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as DashboardData;
}

function parseRisk(json: unknown): RiskMetrics | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as RiskMetrics;
}

export const PortfolioPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const [quickPredict, setQuickPredict] = useState<QuickPredict | null>(null);

  const { data, error, loading, retry } = useFetchWithRetry<DashboardData | null>("/api/v1/monitoring/dashboard", {
    parse: parseDashboard,
    deps: [],
  });

  const riskUrl = `/api/v1/risk/metrics/${primarySymbol}?period=1y`;
  const { data: riskMetrics, error: riskError, loading: riskLoading, retry: riskRetry } = useFetchWithRetry<RiskMetrics | null>(riskUrl, {
    parse: parseRisk,
    deps: [primarySymbol],
  });

  useEffect(() => {
    const fetchQuickPredict = async () => {
      try {
        const res = await fetch(`/api/v1/predictions/quick-predict?symbol=${primarySymbol}`);
        const json = await res.json().catch(() => ({}));
        setQuickPredict(json?.error ? { error: json.error } : json);
      } catch {
        setQuickPredict(null);
      }
    };
    fetchQuickPredict();
    const qId = setInterval(fetchQuickPredict, 60000);
    return () => clearInterval(qId);
  }, [primarySymbol]);

  if (loading) {
    return (
      <section className="panel panel-main-secondary">
        <div className="panel-title">Portfolio & Strategies</div>
        <div className="panel-body-muted">Loading…</div>
      </section>
    );
  }

  if (error) {
    return (
      <PanelErrorState
        sectionClassName="panel panel-main-secondary"
        title="Portfolio & Strategies"
        error={error}
        hint="Start the API on port 8000 and ensure the frontend proxy is used (npm run dev)."
        onRetry={retry}
      />
    );
  }

  const hasModels = (data?.active_models ?? 0) > 0;
  const models = data?.available_models ?? [];
  const recent = data?.recent_predictions ?? [];
  const totalPreds = data?.system?.total_predictions ?? 0;

  const signalColor = (s: number) => (s > 0 ? "var(--accent-green)" : s < 0 ? "var(--accent-red)" : "var(--text-soft)");

  return (
    <section className="panel panel-main-secondary">
      <div className="panel-title">Portfolio & Strategies</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        {quickPredict && !quickPredict.error && (
          <div style={{ marginBottom: 8 }}>
            <span className="num-mono" style={{ color: "var(--accent)" }}>ML Signal ({primarySymbol})</span>
            {" "}
            <span className="num-mono">{quickPredict.recommendation ?? "—"}</span>
            {" "}
            <span className="num-mono" style={{ color: signalColor(quickPredict.signal ?? 0) }}>
              {(quickPredict.signal ?? 0).toFixed(2)}
            </span>
            {quickPredict.current_price != null && (
              <span className="num-mono" style={{ marginLeft: 4 }}>@ ${quickPredict.current_price.toFixed(2)}</span>
            )}
          </div>
        )}
        {!hasModels ? (
          <p className="panel-body-muted">
            No models loaded yet. Train or load models via the API. Dashboard refreshes every 30s.
          </p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <tbody>
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Active models</td>
                <td className="num-mono" style={{ textAlign: "right" }}>{data?.active_models ?? 0}</td>
              </tr>
              {models.length > 0 && (
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Available</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>{models.join(", ")}</td>
                </tr>
              )}
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Total predictions</td>
                <td className="num-mono" style={{ textAlign: "right" }}>{totalPreds}</td>
              </tr>
            </tbody>
          </table>
        )}
        {recent.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>Recent predictions</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Model</th>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Symbol</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Signal</th>
                </tr>
              </thead>
              <tbody>
                {recent.slice(-5).reverse().map((p, i) => (
                  <tr key={i}>
                    <td className="num-mono">{p.model_name ?? "—"}</td>
                    <td className="num-mono" style={{ color: "var(--accent)" }}>{p.symbol ?? "—"}</td>
                    <td className="num-mono" style={{ textAlign: "right", color: signalColor(typeof p.signal === "number" ? p.signal : 0) }}>
                      {typeof p.signal === "number" ? p.signal.toFixed(2) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--accent)", marginBottom: 6 }}>Risk ({primarySymbol})</div>
          {riskLoading && <div className="panel-body-muted" style={{ fontSize: 11 }}>Loading…</div>}
          {!riskLoading && riskError && (
            <div className="panel-body-muted" style={{ fontSize: 11 }}>
              {riskError}
              <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={riskRetry}>Retry</button>
            </div>
          )}
          {!riskLoading && riskMetrics && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <tbody>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>VaR 95%</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>{riskMetrics.var_95_pct != null ? `${riskMetrics.var_95_pct.toFixed(2)}%` : "—"}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>VaR 99%</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>{riskMetrics.var_99_pct != null ? `${riskMetrics.var_99_pct.toFixed(2)}%` : "—"}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>CVaR 95%</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>{riskMetrics.cvar_95_pct != null ? `${riskMetrics.cvar_95_pct.toFixed(2)}%` : "—"}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Vol (ann.)</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>{riskMetrics.volatility_annual_pct != null ? `${riskMetrics.volatility_annual_pct.toFixed(2)}%` : "—"}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Max DD</td>
                  <td className="num-mono" style={{ textAlign: "right", color: "var(--accent-red)" }}>{riskMetrics.max_drawdown_pct != null ? `${riskMetrics.max_drawdown_pct.toFixed(2)}%` : "—"}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Sharpe</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>{riskMetrics.sharpe_ratio != null ? riskMetrics.sharpe_ratio.toFixed(2) : "—"}</td>
                </tr>
              </tbody>
            </table>
          )}
          {!riskLoading && !riskError && !riskMetrics && <div className="panel-body-muted" style={{ fontSize: 11 }}>Risk API unavailable. Start API on port 8000.</div>}
        </div>
      </div>
    </section>
  );
};
