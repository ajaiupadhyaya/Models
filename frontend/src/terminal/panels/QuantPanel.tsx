import React, { useEffect, useState, useCallback } from "react";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
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

const MODEL_TYPES = ["simple", "ensemble", "lstm"] as const;

export const QuantPanel: React.FC = () => {
  const { primarySymbol, lastBacktestSymbol } = useTerminal();
  const { data: modelsList, error: modelsError, loading: modelsLoading, retry: modelsRetry } = useFetchWithRetry<ModelInfo[] | null>("/api/v1/models", {
    parse: parseModels,
  });
  const models = modelsList ?? [];
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compareModels, setCompareModels] = useState<string[]>([]);
  const [compareResult, setCompareResult] = useState<{
    strategies: Array<{ model_name: string; metrics?: Record<string, number | null>; error?: string }>;
    best_strategy?: { model_name: string };
  } | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [wfModel, setWfModel] = useState("");
  const [wfStart, setWfStart] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 2);
    return d.toISOString().slice(0, 10);
  });
  const [wfEnd, setWfEnd] = useState(() => new Date().toISOString().slice(0, 10));
  const [wfTrain, setWfTrain] = useState(252);
  const [wfTest, setWfTest] = useState(63);
  const [wfResult, setWfResult] = useState<Record<string, unknown> | null>(null);
  const [wfLoading, setWfLoading] = useState(false);
  const [wfError, setWfError] = useState<string | null>(null);
  const symbol = lastBacktestSymbol || primarySymbol;
  useEffect(() => { if (models.length > 0 && !wfModel) setWfModel(models[0].name); }, [models, wfModel]);

  // Train form state
  const [trainSymbol, setTrainSymbol] = useState(primarySymbol);
  const [trainModelName, setTrainModelName] = useState("");
  const [trainModelType, setTrainModelType] = useState<"simple" | "ensemble" | "lstm">("ensemble");
  const [trainStart, setTrainStart] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 2);
    return d.toISOString().slice(0, 10);
  });
  const [trainEnd, setTrainEnd] = useState(() => new Date().toISOString().slice(0, 10));
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainResult, setTrainResult] = useState<{ success: boolean; message: string; trainingTime?: number } | null>(null);
  useEffect(() => setTrainSymbol(primarySymbol), [primarySymbol]);

  const runTrain = useCallback(async () => {
    const modelName = trainModelName.trim() || `model_${trainSymbol}_${trainModelType}`;
    setTrainLoading(true);
    setTrainResult(null);
    try {
      const res = await fetch("/api/v1/models/train", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          model_type: trainModelType,
          model_name: modelName,
          symbol: trainSymbol,
          start_date: trainStart,
          end_date: trainEnd,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setTrainResult({ success: false, message: (json?.detail ?? `HTTP ${res.status}`) as string });
        return;
      }
      setTrainResult({
        success: true,
        message: (json?.message ?? "Trained") as string,
        trainingTime: json?.training_time as number | undefined,
      });
      modelsRetry();
    } catch (err) {
      setTrainResult({ success: false, message: err instanceof Error ? err.message : "Request failed" });
    } finally {
      setTrainLoading(false);
    }
  }, [trainSymbol, trainModelName, trainModelType, trainStart, trainEnd, modelsRetry]);

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
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
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

  const runCompare = useCallback(async () => {
    const toCompare = compareModels.length >= 2 ? compareModels : models.slice(0, 2).map((m) => m.name);
    if (toCompare.length < 2) {
      setCompareError("Select at least 2 models to compare.");
      return;
    }
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 1);
    setCompareLoading(true);
    setCompareError(null);
    setCompareResult(null);
    try {
      const res = await fetch("/api/v1/backtest/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          model_names: toCompare,
          symbol,
          start_date: start.toISOString().slice(0, 10),
          end_date: end.toISOString().slice(0, 10),
          initial_capital: 100000,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setCompareError((json?.detail ?? `HTTP ${res.status}`) as string);
        return;
      }
      setCompareResult({
        strategies: (json?.strategies ?? []).map((s: { model_name: string; metrics?: Record<string, number | null>; error?: string }) => ({
          model_name: s.model_name,
          metrics: s.metrics,
          error: s.error,
        })),
        best_strategy: json?.best_strategy,
      });
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setCompareLoading(false);
    }
  }, [compareModels, models, symbol]);

  const runWalkForward = useCallback(async () => {
    const modelName = wfModel || models[0]?.name;
    if (!modelName) {
      setWfError("Select a model.");
      return;
    }
    setWfLoading(true);
    setWfError(null);
    setWfResult(null);
    try {
      const res = await fetch("/api/v1/backtest/walk-forward", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          model_name: modelName,
          symbol,
          start_date: wfStart,
          end_date: wfEnd,
          train_window: wfTrain,
          test_window: wfTest,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setWfError((json?.detail ?? `HTTP ${res.status}`) as string);
        return;
      }
      setWfResult(json as Record<string, unknown>);
    } catch (err) {
      setWfError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setWfLoading(false);
    }
  }, [wfModel, wfStart, wfEnd, wfTrain, wfTest, models, symbol]);

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
        {/* Train model section */}
        <div style={{ marginBottom: 16, paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Train model</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 8 }}>
            <input
              type="text"
              className="ai-input"
              placeholder="Symbol"
              value={trainSymbol}
              onChange={(e) => setTrainSymbol(e.target.value.toUpperCase())}
              style={{ width: 72 }}
              aria-label="Train symbol"
            />
            <input
              type="text"
              className="ai-input"
              placeholder="Model name (optional)"
              value={trainModelName}
              onChange={(e) => setTrainModelName(e.target.value)}
              style={{ width: 120 }}
              aria-label="Model name"
            />
            <select
              value={trainModelType}
              onChange={(e) => setTrainModelType(e.target.value as "simple" | "ensemble" | "lstm")}
              style={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                padding: "6px 8px",
                borderRadius: 4,
                fontSize: 12,
              }}
              aria-label="Model type"
            >
              {MODEL_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <input
              type="date"
              value={trainStart}
              onChange={(e) => setTrainStart(e.target.value)}
              style={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                padding: "4px 6px",
                borderRadius: 4,
                fontSize: 11,
              }}
              aria-label="Start date"
            />
            <span style={{ color: "var(--text-soft)" }}>→</span>
            <input
              type="date"
              value={trainEnd}
              onChange={(e) => setTrainEnd(e.target.value)}
              style={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                padding: "4px 6px",
                borderRadius: 4,
                fontSize: 11,
              }}
              aria-label="End date"
            />
            <button
              type="button"
              className="ai-button"
              disabled={trainLoading}
              onClick={runTrain}
            >
              {trainLoading ? "Training…" : "Train"}
            </button>
          </div>
          {trainResult && (
            <div
              style={{
                color: trainResult.success ? "var(--accent-green)" : "var(--accent-red)",
                fontSize: 11,
              }}
            >
              {trainResult.success
                ? `Success: ${trainResult.message}${trainResult.trainingTime != null ? ` (${trainResult.trainingTime.toFixed(1)}s)` : ""}`
                : trainResult.message}
            </div>
          )}
        </div>

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
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Sharpe ratio</td>
                <td className="num-mono" style={{ textAlign: "right" }}>
                  {sharpe != null ? Number(sharpe).toFixed(3) : "—"}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Max drawdown %</td>
                <td className="num-mono" style={{ textAlign: "right" }}>
                  {maxDd != null ? `${Number(maxDd).toFixed(2)}%` : "—"}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Total return %</td>
                <td className="num-mono" style={{ textAlign: "right" }}>
                  {totalReturn != null ? `${Number(totalReturn).toFixed(2)}%` : "—"}
                </td>
              </tr>
            </tbody>
          </table>
        )}

        {/* Compare strategies */}
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 6, fontWeight: 600 }}>Compare strategies</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 8 }}>
            {models.slice(0, 5).map((m) => (
              <label key={m.name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <input
                  type="checkbox"
                  checked={compareModels.includes(m.name)}
                  onChange={(e) => setCompareModels((prev) => e.target.checked ? [...prev, m.name] : prev.filter((x) => x !== m.name))}
                  style={{ accentColor: "var(--accent)" }}
                />
                <span style={{ fontSize: 11 }}>{m.name}</span>
              </label>
            ))}
            <button type="button" className="ai-button" disabled={compareLoading || models.length < 2} onClick={runCompare}>
              {compareLoading ? "Comparing…" : "Compare"}
            </button>
          </div>
          {compareError && <div style={{ color: "var(--accent-red)", fontSize: 11, marginBottom: 4 }}>{compareError}</div>}
          {compareResult && compareResult.strategies.length > 0 && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Model</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Sharpe</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Max DD %</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Return %</th>
                </tr>
              </thead>
              <tbody>
                {compareResult.strategies.map((s) => (
                  <tr key={s.model_name}>
                    <td style={{ color: "var(--accent)" }}>{s.model_name}</td>
                    {s.error ? (
                      <td colSpan={3} style={{ color: "var(--accent-red)", fontSize: 10 }}>{s.error}</td>
                    ) : (
                      <>
                        <td className="num-mono" style={{ textAlign: "right" }}>
                          {(s.metrics?.sharpe_ratio ?? s.metrics?.sharpe) != null ? Number(s.metrics.sharpe_ratio ?? s.metrics.sharpe).toFixed(3) : "—"}
                        </td>
                        <td className="num-mono" style={{ textAlign: "right" }}>
                          {(s.metrics?.max_drawdown_pct ?? s.metrics?.max_drawdown) != null ? `${Number(s.metrics.max_drawdown_pct ?? s.metrics.max_drawdown).toFixed(2)}%` : "—"}
                        </td>
                        <td className="num-mono" style={{ textAlign: "right" }}>
                          {(s.metrics?.total_return_pct ?? s.metrics?.total_return ?? s.metrics?.cumulative_return) != null ? `${Number(s.metrics.total_return_pct ?? s.metrics.total_return ?? s.metrics.cumulative_return).toFixed(2)}%` : "—"}
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Walk-forward */}
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 6, fontWeight: 600 }}>Walk-forward</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 8 }}>
            <select
              value={wfModel}
              onChange={(e) => setWfModel(e.target.value)}
              style={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                padding: "4px 8px",
                borderRadius: 4,
                fontSize: 12,
              }}
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
            <input type="date" value={wfStart} onChange={(e) => setWfStart(e.target.value)} style={{ background: "var(--bg-panel)", border: "1px solid var(--border)", padding: "4px 6px", borderRadius: 4, fontSize: 11 }} />
            <input type="date" value={wfEnd} onChange={(e) => setWfEnd(e.target.value)} style={{ background: "var(--bg-panel)", border: "1px solid var(--border)", padding: "4px 6px", borderRadius: 4, fontSize: 11 }} />
            <input type="number" value={wfTrain} onChange={(e) => setWfTrain(Number(e.target.value) || 252)} style={{ width: 56, background: "var(--bg-panel)", border: "1px solid var(--border)", padding: "4px 6px", borderRadius: 4, fontSize: 11 }} title="Train window" />
            <input type="number" value={wfTest} onChange={(e) => setWfTest(Number(e.target.value) || 63)} style={{ width: 56, background: "var(--bg-panel)", border: "1px solid var(--border)", padding: "4px 6px", borderRadius: 4, fontSize: 11 }} title="Test window" />
            <button type="button" className="ai-button" disabled={wfLoading || models.length === 0} onClick={runWalkForward}>
              {wfLoading ? "Running…" : "Run walk-forward"}
            </button>
          </div>
          {wfError && <div style={{ color: "var(--accent-red)", fontSize: 11, marginBottom: 4 }}>{wfError}</div>}
          {wfResult && (
            <div style={{ fontSize: 11, color: "var(--text-soft)" }}>
              {wfResult.windows && Array.isArray(wfResult.windows) ? (
                <>Windows: {(wfResult.windows as unknown[]).length}. Avg Sharpe: {(wfResult as { avg_sharpe?: number }).avg_sharpe != null ? Number((wfResult as { avg_sharpe: number }).avg_sharpe).toFixed(3) : "—"}</>
              ) : (
                <>Done. Check response for details.</>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
