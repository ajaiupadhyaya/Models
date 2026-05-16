import React, { useState, useCallback, useEffect } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useTerminal } from "../TerminalContext";
import { TERMINAL_API_ENDPOINTS } from "../endpoints";
import { AreaChart } from "../../charts";
import type { TimeSeriesPoint } from "../../charts";
import { PanelErrorState } from "./PanelErrorState";

interface BacktestMetrics {
  sharpe_ratio?: number;
  sortino_ratio?: number;
  cagr_pct?: number;
  max_drawdown_pct?: number;
  total_return_pct?: number;
  win_rate_pct?: number;
  num_trades?: number;
  alpha_vs_spy?: number;
  beta_vs_spy?: number;
}

interface BacktestResult {
  symbol?: string;
  strategy?: string;
  start_date?: string;
  end_date?: string;
  equity_curve?: Array<{ date: string; equity: number }>;
  trades?: Array<{
    entry_date: string;
    exit_date: string;
    entry_price: number;
    exit_price: number;
    quantity: number;
    pnl: number;
  }>;
  metrics?: BacktestMetrics;
  error?: string;
}

const STRATEGIES = [
  { value: "sma_cross", label: "Moving Average Crossover" },
  { value: "rsi_mean_reversion", label: "RSI Mean Reversion" },
  { value: "factor_momentum", label: "Factor Momentum" },
] as const;

export const BacktestPanel: React.FC = () => {
  const { primarySymbol, lastBacktestSymbol } = useTerminal();
  const [symbol, setSymbol] = useState(primarySymbol);
  const [strategy, setStrategy] = useState<typeof STRATEGIES[number]["value"]>("sma_cross");
  const [startDate, setStartDate] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 3);
    return d.toISOString().slice(0, 10);
  });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [initialCapital] = useState(100000);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (lastBacktestSymbol) {
      setSymbol(lastBacktestSymbol);
      return;
    }
    setSymbol(primarySymbol);
  }, [lastBacktestSymbol, primarySymbol]);

  const runBacktest = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(resolveApiUrl(TERMINAL_API_ENDPOINTS.quantBacktest), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: symbol.toUpperCase(),
          strategy,
          start_date: startDate,
          end_date: endDate,
          initial_capital: initialCapital,
          commission: 0.001,
        }),
      });
      const json = (await res.json().catch(() => ({}))) as BacktestResult;
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
  }, [symbol, strategy, startDate, endDate, initialCapital]);

  const m = result?.metrics;
  const equityData = result?.equity_curve ?? [];

  return (
    <section className="panel panel-main">
      <div className="panel-title">Backtest</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Strategy configuration</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
            <input
              type="text"
              className="ai-input"
              placeholder="Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              style={{ width: 80 }}
              aria-label="Symbol"
            />
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as typeof strategy)}
              style={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                padding: "6px 8px",
                borderRadius: 4,
                fontSize: 12,
              }}
            >
              {STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ background: "var(--bg-panel)", border: "1px solid var(--border)", color: "var(--text)", padding: "4px 6px", borderRadius: 4, fontSize: 11 }}
            />
            <span style={{ color: "var(--text-soft)" }}>→</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ background: "var(--bg-panel)", border: "1px solid var(--border)", color: "var(--text)", padding: "4px 6px", borderRadius: 4, fontSize: 11 }}
            />
            <button
              type="button"
              className="ai-button"
              disabled={loading}
              onClick={runBacktest}
            >
              {loading ? "Running…" : "Run backtest"}
            </button>
          </div>
        </div>

        {error && (
          <PanelErrorState title="Backtest" error={error} onRetry={runBacktest} sectionClassName="panel panel-main-secondary" />
        )}

        {!error && loading && !result && (
          <div className="panel-skeleton" style={{ padding: 24 }}>
            <div className="panel-skeleton-line short" />
            <div className="panel-skeleton-line medium" />
            <div className="panel-skeleton-line" />
            <div className="panel-skeleton-line short" />
          </div>
        )}

        {!error && !loading && !result && (
          <div style={{ color: "var(--text-soft)", padding: 24, textAlign: "center" }}>
            Select a symbol, strategy, and date range, then click &quot;Run backtest&quot;.
          </div>
        )}

        {result && result.equity_curve && equityData.length >= 2 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ color: "var(--accent)", marginBottom: 4, fontSize: 11 }}>Equity curve</div>
            <AreaChart
              data={equityData.map((p) => ({ date: new Date(p.date), value: p.equity })) as TimeSeriesPoint[]}
              height={180}
              marginPreset="compact"
              title=""
              className="chart-root"
              valueFormat={(v) => `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
              xAxisLabel="Date"
              yAxisLabel="Equity ($)"
            />
          </div>
        )}

        {result && m && (
          <>
            <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Metrics</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <tbody>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Sharpe ratio</td><td className="num-mono" style={{ textAlign: "right" }}>{(m.sharpe_ratio ?? 0).toFixed(3)}</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Sortino ratio</td><td className="num-mono" style={{ textAlign: "right" }}>{(m.sortino_ratio ?? 0).toFixed(3)}</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>CAGR %</td><td className="num-mono" style={{ textAlign: "right" }}>{(m.cagr_pct ?? 0).toFixed(2)}%</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Max drawdown %</td><td className="num-mono" style={{ textAlign: "right", color: "var(--accent-red)" }}>{(m.max_drawdown_pct ?? 0).toFixed(2)}%</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Win rate %</td><td className="num-mono" style={{ textAlign: "right" }}>{(m.win_rate_pct ?? 0).toFixed(1)}%</td></tr>
                <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Trades</td><td className="num-mono" style={{ textAlign: "right" }}>{m.num_trades ?? 0}</td></tr>
                {m.alpha_vs_spy != null && <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Alpha vs SPY</td><td className="num-mono" style={{ textAlign: "right" }}>{(m.alpha_vs_spy ?? 0).toFixed(2)}%</td></tr>}
                {m.beta_vs_spy != null && <tr><td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Beta vs SPY</td><td className="num-mono" style={{ textAlign: "right" }}>{(m.beta_vs_spy ?? 0).toFixed(2)}</td></tr>}
              </tbody>
            </table>
          </>
        )}

        {result && result.trades && result.trades.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ color: "var(--text-soft)", marginBottom: 6, fontWeight: 600 }}>Trade log (last 20)</div>
            <div style={{ overflowX: "auto", maxHeight: 200, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Entry</th>
                    <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Exit</th>
                    <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.slice(-20).reverse().map((t, i) => (
                    <tr key={i}>
                      <td style={{ padding: "2px 4px" }}>{t.entry_date} @ {t.entry_price?.toFixed(2)}</td>
                      <td style={{ padding: "2px 4px" }}>{t.exit_date ?? "—"} @ {t.exit_price?.toFixed(2) ?? "—"}</td>
                      <td className="num-mono" style={{ textAlign: "right", color: (t.pnl ?? 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>{(t.pnl ?? 0).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
