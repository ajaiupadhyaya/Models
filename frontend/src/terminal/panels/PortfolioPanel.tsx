import React, { useEffect, useState } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
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

interface StressScenario {
  scenario_id?: string;
  name?: string;
  estimated_return_pct?: number;
}

interface StressResponse {
  ticker?: string;
  scenarios?: StressScenario[];
  error?: string;
}

function parseDashboard(json: unknown): DashboardData | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as DashboardData;
}

function parseRisk(json: unknown): RiskMetrics | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as RiskMetrics;
}

function parseStress(json: unknown): StressResponse | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as StressResponse;
}

interface OptimizeResponse {
  symbols?: string[];
  weights?: Record<string, number>;
  expected_return?: number;
  volatility?: number;
  sharpe_ratio?: number | null;
  error?: string;
}

function parseOptimize(json: unknown): OptimizeResponse | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as OptimizeResponse;
}

function PortfolioOptimizeBlock() {
  const { watchlist } = useTerminal();
  const symbolsParam = watchlist.length >= 2 ? watchlist.slice(0, 10).join(",") : "AAPL,MSFT,GOOGL,AMZN,TSLA";
  const url = `/api/v1/risk/optimize?symbols=${encodeURIComponent(symbolsParam)}&period=1y&method=sharpe`;
  const { data, error: optError, loading: optLoading, retry: optRetry } = useFetchWithRetry<OptimizeResponse | null>(url, {
    parse: parseOptimize,
    deps: [symbolsParam],
  });
  if (optLoading) return <div className="panel-body-muted" style={{ fontSize: 11 }}>Loading…</div>;
  if (optError || data?.error) return <div className="panel-body-muted" style={{ fontSize: 11 }}>{data?.error ?? optError} <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={optRetry}>Retry</button></div>;
  const weights = data?.weights ?? {};
  const entries = Object.entries(weights).filter(([, v]) => typeof v === "number");
  if (entries.length === 0) return <div className="panel-body-muted" style={{ fontSize: 11 }}>Need at least 2 symbols in watchlist for optimization.</div>;
  return (
    <div style={{ fontSize: 11 }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          {entries.map(([sym, w]) => (
            <tr key={sym}>
              <td className="num-mono" style={{ color: "var(--accent)", padding: "2px 8px 2px 0" }}>{sym}</td>
              <td className="num-mono" style={{ textAlign: "right" }}>{(Number(w) * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      {(data?.sharpe_ratio != null || data?.volatility != null) && (
        <div style={{ marginTop: 6, color: "var(--text-soft)" }}>
          {data.sharpe_ratio != null && <span className="num-mono">Sharpe {data.sharpe_ratio.toFixed(2)}</span>}
          {data.volatility != null && <span className="num-mono" style={{ marginLeft: 8 }}>Vol {(data.volatility * 100).toFixed(2)}%</span>}
        </div>
      )}
    </div>
  );
}

interface ReportsHealth {
  status?: string;
  openai_configured?: boolean;
}

function InvestorReportsBlock() {
  const { data } = useFetchWithRetry<ReportsHealth | null>("/api/v1/reports/health", {
    parse: (json) => (json && typeof json === "object" && !("detail" in (json as object)) ? (json as ReportsHealth) : null),
  });
  const base = typeof window !== "undefined" ? window.location.origin : "";
  const docsUrl = `${base}/docs`;
  return (
    <div style={{ fontSize: 11, color: "var(--text-soft)" }}>
      {data?.status === "healthy" ? (
        <>
          <span style={{ color: "var(--accent-green)" }}>Available</span>
          {data.openai_configured ? " (OpenAI configured)." : " (Set OPENAI_API_KEY for generation)."}
        </>
      ) : (
        <span>Check API /docs for report generation.</span>
      )}
      {" "}
      <a href={docsUrl} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)" }}>API docs</a>
    </div>
  );
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

  const stressUrl = `/api/v1/risk/stress?ticker=${encodeURIComponent(primarySymbol)}`;
  const { data: stressData, error: stressError, loading: stressLoading, retry: stressRetry } = useFetchWithRetry<StressResponse | null>(stressUrl, {
    parse: parseStress,
    deps: [primarySymbol],
  });

  useEffect(() => {
    const fetchQuickPredict = async () => {
      try {
        const res = await fetch(resolveApiUrl(`/api/v1/predictions/quick-predict?symbol=${primarySymbol}`), { headers: getAuthHeaders() });
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
        <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--accent)", marginBottom: 6 }}>Stress testing ({primarySymbol})</div>
          {stressLoading && <div className="panel-body-muted" style={{ fontSize: 11 }}>Loading…</div>}
          {!stressLoading && stressError && (
            <div className="panel-body-muted" style={{ fontSize: 11 }}>
              {stressError}
              <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={stressRetry}>Retry</button>
            </div>
          )}
          {!stressLoading && stressData?.scenarios && stressData.scenarios.length > 0 && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Scenario</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Est. return %</th>
                </tr>
              </thead>
              <tbody>
                {stressData.scenarios.map((s, i) => (
                  <tr key={i}>
                    <td style={{ color: "var(--text)", padding: "2px 8px 2px 0" }}>{s.name ?? s.scenario_id ?? "—"}</td>
                    <td className="num-mono" style={{ textAlign: "right", color: (s.estimated_return_pct ?? 0) < 0 ? "var(--accent-red)" : "var(--text)" }}>
                      {s.estimated_return_pct != null ? `${s.estimated_return_pct.toFixed(2)}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!stressLoading && stressData && (!stressData.scenarios || stressData.scenarios.length === 0) && !stressError && (
            <div className="panel-body-muted" style={{ fontSize: 11 }}>No stress scenarios available.</div>
          )}
        </div>
        <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--accent)", marginBottom: 6 }}>Portfolio optimization (Max Sharpe)</div>
          <PortfolioOptimizeBlock />
        </div>
        <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--accent)", marginBottom: 6 }}>Investor reports</div>
          <InvestorReportsBlock />
        </div>
      </div>
    </section>
  );
};
