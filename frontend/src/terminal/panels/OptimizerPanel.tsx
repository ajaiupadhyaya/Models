import React, { useState, useCallback } from "react";
import { resolveApiUrl } from "../../apiBase";
import { BarChart, XyLineChart } from "../../charts";
import { PanelErrorState } from "./PanelErrorState";

interface OptimizerResult {
  tickers?: string[];
  weights?: Record<string, number>;
  expected_return?: number;
  volatility?: number;
  sharpe_ratio?: number;
  efficient_frontier?: Array<{ return: number; volatility: number; sharpe: number }>;
  period?: { start: string; end: string };
  error?: string;
}

const DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,V";

export const OptimizerPanel: React.FC = () => {
  const [tickersInput, setTickersInput] = useState(DEFAULT_TICKERS);
  const [result, setResult] = useState<OptimizerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runOptimize = useCallback(async () => {
    const tickers = tickersInput.split(/[\s,]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (tickers.length < 2) {
      setError("Provide at least 2 tickers");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const end = new Date().toISOString().slice(0, 10);
      const start = new Date();
      start.setFullYear(start.getFullYear() - 5);
      const res = await fetch(resolveApiUrl("/api/v1/risk/optimize"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers,
          start_date: start.toISOString().slice(0, 10),
          end_date: end,
          risk_free_rate: 0.02,
        }),
      });
      const json = (await res.json().catch(() => ({}))) as OptimizerResult;
      if (!res.ok) {
        setError((json as { detail?: string }).detail ?? json.error ?? `HTTP ${res.status}`);
        return;
      }
      if (json.error) {
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
      <div className="panel-title">Portfolio Optimizer</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Tickers (comma-separated)</div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <input
              type="text"
              className="ai-input"
              placeholder="AAPL, MSFT, GOOGL, …"
              value={tickersInput}
              onChange={(e) => setTickersInput(e.target.value)}
              style={{ flex: 1, minWidth: 200 }}
            />
            <button type="button" className="ai-button" disabled={loading} onClick={runOptimize}>
              {loading ? "Optimizing…" : "Optimize"}
            </button>
          </div>
        </div>

        {error && (
          <PanelErrorState title="Optimizer" error={error} onRetry={runOptimize} sectionClassName="panel panel-main-secondary" />
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
            Enter 2+ tickers, then click &quot;Optimize&quot;. Uses real historical returns from DB.
          </div>
        )}

        {result && result.weights && Object.keys(result.weights).length > 0 && (
          <>
            <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Optimal weights</div>
            <BarChart
              data={Object.entries(result.weights).map(([k, v]) => ({
                label: k,
                value: (v ?? 0) * 100,
                color: "var(--accent)",
              }))}
              height={Math.min(200, Object.keys(result.weights).length * 28)}
              marginPreset="compact"
              horizontal
              valueFormat={(v) => `${v.toFixed(1)}%`}
              title="Weight %"
              className="chart-root"
            />
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, marginTop: 12 }}>
              <tbody>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Expected return</td><td className="num-mono" style={{ textAlign: "right" }}>{((result.expected_return ?? 0) * 100).toFixed(2)}%</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Volatility</td><td className="num-mono" style={{ textAlign: "right" }}>{((result.volatility ?? 0) * 100).toFixed(2)}%</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Sharpe ratio</td><td className="num-mono" style={{ textAlign: "right" }}>{(result.sharpe_ratio ?? 0).toFixed(3)}</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Period</td><td style={{ textAlign: "right" }}>{result.period?.start ?? ""} → {result.period?.end ?? ""}</td></tr>
              </tbody>
            </table>
            {result.efficient_frontier && result.efficient_frontier.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ color: "var(--text-soft)", marginBottom: 6, fontWeight: 600 }}>Efficient frontier</div>
                <XyLineChart
                  data={result.efficient_frontier.map((p, i) => ({
                    x: (p.volatility ?? 0) * 100,
                    y: (p.return ?? 0) * 100,
                    label: `Vol ${((p.volatility ?? 0) * 100).toFixed(2)}%, Return ${((p.return ?? 0) * 100).toFixed(2)}%, Sharpe ${(p.sharpe ?? 0).toFixed(2)}`,
                  }))}
                  height={160}
                  marginPreset="compact"
                  xFormat={(v) => `${v.toFixed(2)}%`}
                  yFormat={(v) => `${v.toFixed(2)}%`}
                  xAxisLabel="Volatility (%)"
                  yAxisLabel="Return (%)"
                  className="chart-root"
                />
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
};
