import React, { useEffect, useState } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";
import { BarChart } from "../../charts";

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
  if (optLoading) return <div className="text-on-surface-variant font-data-mono text-[10px] animate-pulse">Loading…</div>;
  if (optError || data?.error) return <div className="text-error font-data-mono text-[10px] flex items-center gap-2">{data?.error ?? optError} <button type="button" className="border border-error px-2 py-1" onClick={optRetry}>RETRY</button></div>;
  const weights = data?.weights ?? {};
  const entries = Object.entries(weights).filter(([, v]) => typeof v === "number");
  if (entries.length === 0) return <div className="text-on-surface-variant font-data-mono text-[10px] uppercase">Need at least 2 symbols in watchlist for optimization.</div>;
  const allocationData = entries.map(([sym, w]) => ({ label: sym, value: Number(w) * 100 }));
  return (
    <div className="font-data-mono text-[12px]">
      <div className="mb-4">
        <BarChart
          data={allocationData}
          height={Math.min(140, allocationData.length * 24)}
          marginPreset="compact"
          horizontal
          valueFormat={(v) => `${v.toFixed(1)}%`}
          className="w-full"
        />
      </div>
      <table className="w-full border-collapse">
        <tbody>
          {entries.map(([sym, w]) => (
            <tr key={sym} className="border-b border-outline-variant/50 hover:bg-background/50">
              <td className="text-primary py-1">{sym}</td>
              <td className="text-right py-1">{(Number(w) * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      {(data?.sharpe_ratio != null || data?.volatility != null) && (
        <div className="mt-2 text-on-surface-variant flex gap-4 uppercase">
          {data.sharpe_ratio != null && <span>Sharpe <span className="text-on-surface">{data.sharpe_ratio.toFixed(2)}</span></span>}
          {data.volatility != null && <span>Vol <span className="text-on-surface">{(data.volatility * 100).toFixed(2)}%</span></span>}
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
    <div className="font-data-mono text-[12px] text-on-surface-variant uppercase flex flex-col gap-1">
      <div>
        {data?.status === "healthy" ? (
          <>
            <span className="text-accent-green">Available</span>
            {data.openai_configured ? " (OpenAI configured)." : " (Set OPENAI_API_KEY for generation)."}
          </>
        ) : (
          <span>See <a href={docsUrl} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">/docs</a> for report generation.</span>
        )}
      </div>
      <div>
        <a href={docsUrl} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">API docs</a>
      </div>
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
      <div className="flex flex-col h-full bg-surface-container-low text-on-surface p-4">
        <div className="font-label-xs text-label-xs uppercase text-on-tertiary-container tracking-[0.4em] mb-4">PORTFOLIO & STRATEGIES</div>
        <div className="text-on-surface-variant font-data-mono text-[12px] animate-pulse">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-full bg-surface-container-low text-on-surface p-4">
        <div className="font-label-xs text-label-xs uppercase text-on-tertiary-container tracking-[0.4em] mb-4">PORTFOLIO & STRATEGIES</div>
        <div className="text-error font-data-mono text-[12px]">{error}</div>
        <button type="button" className="border border-error px-2 py-1 mt-2 text-error font-data-mono text-[10px]" onClick={retry}>RETRY</button>
      </div>
    );
  }

  const hasModels = (data?.active_models ?? 0) > 0;
  const models = data?.available_models ?? [];
  const recent = data?.recent_predictions ?? [];
  const totalPreds = data?.system?.total_predictions ?? 0;

  const signalColorClass = (s: number) => (s > 0 ? "text-accent-green" : s < 0 ? "text-error" : "text-on-surface-variant");

  return (
    <div className="flex flex-col h-full bg-surface-container-low text-on-surface overflow-y-auto">
      <div className="font-label-xs text-label-xs uppercase text-on-tertiary-container tracking-[0.4em] mb-4">PORTFOLIO & STRATEGIES</div>
      <div className="font-data-mono text-[12px]">
        {quickPredict && !quickPredict.error && (
          <div className="mb-4 bg-background p-2 border border-outline-variant">
            <span className="text-primary font-bold uppercase">ML Signal ({primarySymbol})</span>
            {" "}
            <span className="uppercase ml-2">{quickPredict.recommendation ?? "—"}</span>
            {" "}
            <span className={`ml-2 ${signalColorClass(quickPredict.signal ?? 0)}`}>
              {(quickPredict.signal ?? 0).toFixed(2)}
            </span>
            {quickPredict.current_price != null && (
              <span className="ml-2">@ ${quickPredict.current_price.toFixed(2)}</span>
            )}
          </div>
        )}
        {!hasModels ? (
          <p className="text-on-surface-variant uppercase text-[10px]">
            No models loaded yet. Train or load models via the API. Dashboard refreshes every 30s.
          </p>
        ) : (
          <table className="w-full border-collapse mb-4">
            <tbody>
              <tr className="border-b border-outline-variant">
                <td className="text-on-surface-variant py-1">Active models</td>
                <td className="text-right font-medium">{data?.active_models ?? 0}</td>
              </tr>
              {models.length > 0 && (
                <tr className="border-b border-outline-variant">
                  <td className="text-on-surface-variant py-1">Available</td>
                  <td className="text-right font-medium">{models.join(", ")}</td>
                </tr>
              )}
              <tr>
                <td className="text-on-surface-variant py-1">Total predictions</td>
                <td className="text-right font-medium">{totalPreds}</td>
              </tr>
            </tbody>
          </table>
        )}
        {recent.length > 0 && (
          <div className="mt-4">
            <div className="text-primary uppercase mb-2">Recent predictions</div>
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-outline-variant text-on-surface-variant">
                  <th className="text-left font-normal py-1">Model</th>
                  <th className="text-left font-normal py-1">Symbol</th>
                  <th className="text-right font-normal py-1">Signal</th>
                </tr>
              </thead>
              <tbody>
                {recent.slice(-5).reverse().map((p, i) => (
                  <tr key={i} className="border-b border-outline-variant/50 hover:bg-background/50">
                    <td className="py-1">{p.model_name ?? "—"}</td>
                    <td className="py-1 text-primary">{p.symbol ?? "—"}</td>
                    <td className={`py-1 text-right ${signalColorClass(typeof p.signal === "number" ? p.signal : 0)}`}>
                      {typeof p.signal === "number" ? p.signal.toFixed(2) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="mt-6 pt-4 border-t border-outline-variant">
          <div className="text-primary uppercase mb-2">Risk ({primarySymbol})</div>
          {riskLoading && <div className="text-on-surface-variant animate-pulse">Loading…</div>}
          {!riskLoading && riskError && (
            <div className="text-error flex items-center gap-2">
              {riskError}
              <button type="button" className="border border-error px-2 py-1 text-[10px]" onClick={riskRetry}>RETRY</button>
            </div>
          )}
          {!riskLoading && riskMetrics && (
            <>
              <div className="mb-4">
                <BarChart
                  data={[
                    { label: "VaR 95%", value: riskMetrics.var_95_pct ?? 0, color: "var(--accent-red)" },
                    { label: "VaR 99%", value: riskMetrics.var_99_pct ?? 0, color: "var(--accent-red)" },
                    { label: "CVaR 95%", value: riskMetrics.cvar_95_pct ?? 0, color: "var(--accent-red)" },
                    { label: "Vol %", value: riskMetrics.volatility_annual_pct ?? 0 },
                    { label: "Max DD %", value: riskMetrics.max_drawdown_pct ?? 0, color: "var(--accent-red)" },
                    { label: "Sharpe", value: riskMetrics.sharpe_ratio ?? 0, color: "var(--accent-green)" },
                  ]}
                  height={140}
                  marginPreset="compact"
                  valueFormat={(v) => (Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2))}
                  className="w-full"
                />
              </div>
              <table className="w-full border-collapse">
                <tbody>
                  <tr className="border-b border-outline-variant/50 hover:bg-background/50">
                    <td className="text-on-surface-variant py-1">VaR 95%</td>
                    <td className="text-right">{riskMetrics.var_95_pct != null ? `${riskMetrics.var_95_pct.toFixed(2)}%` : "—"}</td>
                  </tr>
                  <tr className="border-b border-outline-variant/50 hover:bg-background/50">
                    <td className="text-on-surface-variant py-1">VaR 99%</td>
                    <td className="text-right">{riskMetrics.var_99_pct != null ? `${riskMetrics.var_99_pct.toFixed(2)}%` : "—"}</td>
                  </tr>
                  <tr className="border-b border-outline-variant/50 hover:bg-background/50">
                    <td className="text-on-surface-variant py-1">CVaR 95%</td>
                    <td className="text-right">{riskMetrics.cvar_95_pct != null ? `${riskMetrics.cvar_95_pct.toFixed(2)}%` : "—"}</td>
                  </tr>
                  <tr className="border-b border-outline-variant/50 hover:bg-background/50">
                    <td className="text-on-surface-variant py-1">Vol (ann.)</td>
                    <td className="text-right">{riskMetrics.volatility_annual_pct != null ? `${riskMetrics.volatility_annual_pct.toFixed(2)}%` : "—"}</td>
                  </tr>
                  <tr className="border-b border-outline-variant/50 hover:bg-background/50">
                    <td className="text-on-surface-variant py-1">Max DD</td>
                    <td className="text-right text-error">{riskMetrics.max_drawdown_pct != null ? `${riskMetrics.max_drawdown_pct.toFixed(2)}%` : "—"}</td>
                  </tr>
                  <tr className="hover:bg-background/50">
                    <td className="text-on-surface-variant py-1">Sharpe</td>
                    <td className="text-right">{riskMetrics.sharpe_ratio != null ? riskMetrics.sharpe_ratio.toFixed(2) : "—"}</td>
                  </tr>
                </tbody>
              </table>
            </>
          )}
          {!riskLoading && !riskError && !riskMetrics && <div className="text-on-surface-variant text-[10px] uppercase">Risk data unavailable. Try again or retry.</div>}
        </div>
        <div className="mt-6 pt-4 border-t border-outline-variant">
          <div className="text-primary uppercase mb-2">Stress testing ({primarySymbol})</div>
          {stressLoading && <div className="text-on-surface-variant animate-pulse">Loading…</div>}
          {!stressLoading && stressError && (
            <div className="text-error flex items-center gap-2">
              {stressError}
              <button type="button" className="border border-error px-2 py-1 text-[10px]" onClick={stressRetry}>RETRY</button>
            </div>
          )}
          {!stressLoading && stressData?.scenarios && stressData.scenarios.length > 0 && (
            <>
              <div className="mb-4">
                <BarChart
                  data={stressData.scenarios.map((s) => ({
                    label: (s.name ?? s.scenario_id ?? "—").slice(0, 12),
                    value: s.estimated_return_pct ?? 0,
                    color: (s.estimated_return_pct ?? 0) < 0 ? "var(--accent-red)" : "var(--accent-green)",
                  }))}
                  height={Math.min(120, stressData.scenarios.length * 28)}
                  marginPreset="compact"
                  horizontal
                  valueFormat={(v) => `${v.toFixed(1)}%`}
                  className="w-full"
                />
              </div>
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-outline-variant text-on-surface-variant">
                    <th className="text-left font-normal py-1">Scenario</th>
                    <th className="text-right font-normal py-1">Est. return %</th>
                  </tr>
                </thead>
                <tbody>
                  {stressData.scenarios.map((s, i) => (
                    <tr key={i} className="border-b border-outline-variant/50 hover:bg-background/50">
                      <td className="py-1 text-on-surface">{s.name ?? s.scenario_id ?? "—"}</td>
                      <td className={`py-1 text-right ${(s.estimated_return_pct ?? 0) < 0 ? "text-error" : "text-on-surface"}`}>
                        {s.estimated_return_pct != null ? `${s.estimated_return_pct.toFixed(2)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {!stressLoading && stressData && (!stressData.scenarios || stressData.scenarios.length === 0) && !stressError && (
            <div className="text-on-surface-variant text-[10px] uppercase">No stress scenarios available.</div>
          )}
        </div>
        <div className="mt-6 pt-4 border-t border-outline-variant">
          <div className="text-primary uppercase mb-2">Portfolio optimization (Max Sharpe)</div>
          <PortfolioOptimizeBlock />
        </div>
        <div className="mt-6 pt-4 border-t border-outline-variant">
          <div className="text-primary uppercase mb-2">Investor reports</div>
          <InvestorReportsBlock />
        </div>
      </div>
    </div>
  );
};
