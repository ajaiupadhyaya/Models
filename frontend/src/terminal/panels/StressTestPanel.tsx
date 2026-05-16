import React, { useState, useCallback } from "react";
import { resolveApiUrl } from "../../apiBase";
import { BarChart } from "../../charts";
import { PanelErrorState } from "./PanelErrorState";

interface ScenarioResult {
  scenario_id: string;
  name: string;
  description?: string;
  start?: string;
  end?: string;
  portfolio_drawdown_pct?: number | null;
  worst_single_day_loss_pct?: number | null;
  recovery_days?: number | null;
  error?: string;
}

interface StressTestResult {
  tickers?: string[];
  weights?: Record<string, number>;
  scenarios?: ScenarioResult[];
  error?: string;
}

const DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN,META";

export const StressTestPanel: React.FC = () => {
  const [tickersInput, setTickersInput] = useState(DEFAULT_TICKERS);
  const [result, setResult] = useState<StressTestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runStressTest = useCallback(async () => {
    const tickers = tickersInput.split(/[\s,]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (tickers.length < 1) {
      setError("Provide at least 1 ticker");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(resolveApiUrl("/api/v1/risk/stress-test"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers }),
      });
      const json = (await res.json().catch(() => ({}))) as StressTestResult;
      if (!res.ok) {
        setError((json as { detail?: string }).detail ?? json.error ?? `HTTP ${res.status}`);
        return;
      }
      if (json.error && !json.scenarios?.length) {
        setError(json.error);
        return;
      }
      setResult(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, [tickersInput]);

  return (
    <section className="panel panel-main">
      <div className="panel-title">Stress Testing</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Portfolio tickers</div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <input
              type="text"
              className="ai-input"
              placeholder="AAPL, MSFT, GOOGL, …"
              value={tickersInput}
              onChange={(e) => setTickersInput(e.target.value)}
              style={{ flex: 1, minWidth: 200 }}
            />
            <button type="button" className="ai-button" disabled={loading} onClick={runStressTest}>
              {loading ? "Running…" : "Run stress test"}
            </button>
          </div>
        </div>

        {error && (
          <PanelErrorState title="Stress test" error={error} onRetry={runStressTest} sectionClassName="panel panel-main-secondary" />
        )}

        {!error && loading && !result && (
          <div className="panel-skeleton" style={{ padding: 24 }}>
            <div className="panel-skeleton-line short" />
            <div className="panel-skeleton-line medium" />
            <div className="panel-skeleton-line" />
          </div>
        )}

        {!error && !loading && !result && (
          <div style={{ color: "var(--text-soft)", padding: 24, textAlign: "center" }}>
            Enter portfolio tickers, then click &quot;Run stress test&quot;. Uses real historical drawdowns: 2008 Crisis, COVID, Dot-com, 2022 Rate Shock.
          </div>
        )}

        {result && result.scenarios && result.scenarios.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Crisis scenarios (real historical data)</div>
            {result.scenarios.some((s) => s.portfolio_drawdown_pct != null) && (
              <div style={{ marginBottom: 12 }}>
                <BarChart
                  data={result.scenarios
                    .filter((s) => s.portfolio_drawdown_pct != null)
                    .map((s) => ({
                      label: s.name,
                      value: -(s.portfolio_drawdown_pct ?? 0),
                      color: "var(--accent-red)",
                    }))}
                  height={Math.min(200, result.scenarios.length * 28)}
                  marginPreset="compact"
                  valueFormat={(v) => `${(-v).toFixed(2)}%`}
                  xAxisLabel="Scenario"
                  yAxisLabel="Drawdown %"
                  horizontal
                  className="chart-root"
                />
              </div>
            )}
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Scenario</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Drawdown %</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Worst day %</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Recovery days</th>
                </tr>
              </thead>
              <tbody>
                {result.scenarios.map((s) => (
                  <tr key={s.scenario_id}>
                    <td style={{ padding: "4px 8px 4px 0" }}>
                      <span style={{ color: "var(--accent)" }}>{s.name}</span>
                      {s.description && <span style={{ color: "var(--text-soft)", fontSize: 10, marginLeft: 4 }}>({s.description})</span>}
                    </td>
                    <td className="num-mono" style={{ textAlign: "right", color: "var(--accent-red)" }}>
                      {s.portfolio_drawdown_pct != null ? `${s.portfolio_drawdown_pct.toFixed(2)}%` : s.error ?? "—"}
                    </td>
                    <td className="num-mono" style={{ textAlign: "right", color: "var(--accent-red)" }}>
                      {s.worst_single_day_loss_pct != null ? `${s.worst_single_day_loss_pct.toFixed(2)}%` : "—"}
                    </td>
                    <td className="num-mono" style={{ textAlign: "right" }}>
                      {s.recovery_days != null ? String(s.recovery_days) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};
