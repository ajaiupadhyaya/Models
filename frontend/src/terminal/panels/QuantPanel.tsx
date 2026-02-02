import React, { useEffect, useState, useCallback } from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";

interface ModelInfo {
  name: string;
  type?: string;
  symbol?: string;
  status?: string;
}

interface BacktestResult {
  model_name: string;
  symbol: string;
  period: { start: string; end: string };
  metrics: Record<string, number | null>;
  status?: string;
}

interface ModelsResponse {
  detail?: unknown;
  models?: ModelInfo[];
}

function parseModels(json: unknown): ModelInfo[] | null {
  const r = json as ModelsResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.models) ? (r.models as ModelInfo[]) : [];
}

export const QuantPanel: React.FC = () => {
  const { primarySymbol, lastBacktestSymbol } = useTerminal();
  const { data: modelsList, error: modelsError, loading: modelsLoading, retry: modelsRetry } = useFetchWithRetry<ModelInfo[] | null>("/api/v1/models", {
    parse: parseModels,
  });
  const models = modelsList ?? [];
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const symbol = lastBacktestSymbol || primarySymbol;

  const runBacktest = useCallback(async () => {
    const modelName = models[0]?.name ?? "default";
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 1);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/backtest/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: modelName,
          symbol,
          start_date: start.toISOString().slice(0, 10),
          end_date: end.toISOString().slice(0, 10),
          initial_capital: 100000,
          commission: 0.001,
          position_size: 0.2,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(json?.detail ?? `HTTP ${res.status}`);
        setBacktest(null);
        return;
      }
      setBacktest(json as BacktestResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      setBacktest(null);
    } finally {
      setLoading(false);
    }
  }, [models, symbol]);

  useEffect(() => {
    if (lastBacktestSymbol && models.length > 0 && !backtest && !loading) {
      runBacktest();
    }
  }, [lastBacktestSymbol, models.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const metrics = backtest?.metrics ?? {};
  const sharpe = metrics.sharpe_ratio ?? metrics.sharpe ?? null;
  const maxDd = metrics.max_drawdown_pct ?? metrics.max_drawdown ?? null;
  const totalReturn = metrics.total_return_pct ?? metrics.total_return ?? metrics.cumulative_return ?? null;

  return (
    <section className="panel panel-main">
      <div className="panel-title">Quant & Backtesting</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 8 }}>
          <span style={{ color: "var(--text-soft)" }}>Models: </span>
          {modelsError ? (
            <>
              {modelsError}
              <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={modelsRetry}>Retry</button>
            </>
          ) : modelsLoading && models.length === 0 ? (
            "Loading…"
          ) : models.length === 0 ? (
            "None loaded. Train or load via API."
          ) : (
            models.map((m) => m.name).join(", ")
          )}
        </div>
        <button
          type="button"
          className="ai-button"
          disabled={loading || models.length === 0}
          onClick={runBacktest}
          style={{ marginBottom: 12 }}
        >
          {loading ? "Running…" : `Run backtest (${symbol})`}
        </button>
        {error && (
          <div style={{ color: "var(--accent-red)", marginBottom: 8 }}>{error}</div>
        )}
        {backtest && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <tbody>
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Sharpe</td>
                <td className="num-mono" style={{ textAlign: "right" }}>
                  {sharpe != null ? Number(sharpe).toFixed(3) : "—"}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Max drawdown</td>
                <td className="num-mono" style={{ textAlign: "right" }}>
                  {maxDd != null ? `${Number(maxDd).toFixed(2)}%` : "—"}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Total return</td>
                <td className="num-mono" style={{ textAlign: "right" }}>
                  {totalReturn != null ? `${Number(totalReturn).toFixed(2)}%` : "—"}
                </td>
              </tr>
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
};
