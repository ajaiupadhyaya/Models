import React, { useState, useCallback } from "react";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { PanelErrorState } from "./PanelErrorState";

interface StatusData {
  detail?: unknown;
  status?: string;
  models_initialized?: boolean;
  symbols?: string[];
  last_run?: string;
  error?: string;
  [key: string]: unknown;
}

interface SignalItem {
  symbol?: string;
  action?: string;
  confidence?: number;
  price?: number;
  timestamp?: string;
}

interface SignalsResponse {
  detail?: unknown;
  signals?: SignalItem[];
  count?: number;
}

interface TradeItem {
  symbol?: string;
  side?: string;
  quantity?: number;
  price?: number;
  timestamp?: string;
  [key: string]: unknown;
}

interface TradesResponse {
  detail?: unknown;
  trades?: TradeItem[];
  count?: number;
}

function parseStatus(json: unknown): StatusData | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as StatusData;
}

function parseSignals(json: unknown): SignalItem[] | null {
  const r = json as SignalsResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.signals) ? r.signals : [];
}

function parseTrades(json: unknown): TradeItem[] | null {
  const r = json as TradesResponse;
  if (r?.detail) return null;
  return Array.isArray(r?.trades) ? r.trades : [];
}

export const AutomationPanel: React.FC = () => {
  const [executeTrades, setExecuteTrades] = useState(false);
  const [runLoading, setRunLoading] = useState(false);
  const [runResult, setRunResult] = useState<string | null>(null);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [retrainResult, setRetrainResult] = useState<string | null>(null);

  const { data: statusData, error: statusError, loading: statusLoading, retry: statusRetry } = useFetchWithRetry<StatusData | null>(
    "/api/v1/orchestrator/status",
    { parse: parseStatus }
  );
  const { data: signalsList, error: signalsError } = useFetchWithRetry<SignalItem[] | null>(
    "/api/v1/orchestrator/signals?limit=5",
    { parse: parseSignals }
  );
  const { data: tradesList, error: tradesError } = useFetchWithRetry<TradeItem[] | null>(
    "/api/v1/orchestrator/trades?limit=10",
    { parse: parseTrades }
  );

  const runCycle = useCallback(async () => {
    setRunLoading(true);
    setRunResult(null);
    try {
      const res = await fetch(`/api/v1/orchestrator/run-cycle?execute_trades=${executeTrades}`, {
        method: "POST",
        headers: getAuthHeaders(),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setRunResult((json?.detail ?? `HTTP ${res.status}`) as string);
        return;
      }
      setRunResult(`Cycle completed. ${json?.message ?? ""}`);
      statusRetry();
    } catch (err) {
      setRunResult(err instanceof Error ? err.message : "Request failed");
    } finally {
      setRunLoading(false);
    }
  }, [executeTrades, statusRetry]);

  const retrainModels = useCallback(async () => {
    setRetrainLoading(true);
    setRetrainResult(null);
    try {
      const res = await fetch("/api/v1/orchestrator/retrain", { method: "POST", headers: getAuthHeaders() });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setRetrainResult((json?.detail ?? `HTTP ${res.status}`) as string);
        return;
      }
      setRetrainResult("Retraining started.");
      statusRetry();
    } catch (err) {
      setRetrainResult(err instanceof Error ? err.message : "Request failed");
    } finally {
      setRetrainLoading(false);
    }
  }, [statusRetry]);

  if (statusLoading && !statusData) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Automation / Orchestrator</div>
        <div className="panel-body-muted">Loading…</div>
      </section>
    );
  }

  if (statusError && !statusData) {
    return (
      <PanelErrorState
        title="Automation / Orchestrator"
        error={statusError}
        hint="Orchestrator API may not be loaded. Ensure dependencies (e.g. schedule, stable-baselines3) are installed. See API /info for capabilities."
        onRetry={statusRetry}
      />
    );
  }

  const status = statusData?.status ?? "unknown";
  const modelsInit = statusData?.models_initialized ?? false;
  const symbols = (statusData?.symbols as string[] | undefined) ?? [];

  return (
    <section className="panel panel-main">
      <div className="panel-title">Automation / Orchestrator</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12 }}>
          <span style={{ color: "var(--text-soft)" }}>Status: </span>
          <span style={{ color: modelsInit ? "var(--accent-green)" : "var(--text-soft)" }}>
            {status} {modelsInit ? "(models initialized)" : ""}
          </span>
          {symbols.length > 0 && (
            <span style={{ color: "var(--text-soft)", marginLeft: 8 }}>Symbols: {symbols.join(", ")}</span>
          )}
          <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={statusRetry}>
            Refresh
          </button>
        </div>

        <div style={{ marginBottom: 12, paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 6 }}>Actions</div>
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <input
              type="checkbox"
              checked={executeTrades}
              onChange={(e) => setExecuteTrades(e.target.checked)}
              style={{ accentColor: "var(--accent)" }}
            />
            <span style={{ color: "var(--text)" }}>Execute trades when running cycle</span>
          </label>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              type="button"
              className="ai-button"
              disabled={runLoading}
              onClick={runCycle}
            >
              {runLoading ? "Running…" : "Run cycle"}
            </button>
            <button
              type="button"
              className="ai-button"
              disabled={retrainLoading}
              onClick={retrainModels}
            >
              {retrainLoading ? "Retraining…" : "Retrain models"}
            </button>
          </div>
          {runResult && <div style={{ fontSize: 11, color: "var(--text-soft)", marginTop: 6 }}>{runResult}</div>}
          {retrainResult && <div style={{ fontSize: 11, color: "var(--text-soft)", marginTop: 6 }}>{retrainResult}</div>}
        </div>

        {signalsList && signalsList.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ color: "var(--text-soft)", marginBottom: 4 }}>Recent signals</div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11 }}>
              {signalsList.slice(0, 5).map((s, i) => (
                <li key={i} style={{ marginBottom: 2 }}>
                  {s.symbol} {s.action} @ {s.price != null ? Number(s.price).toFixed(2) : "—"} (conf: {s.confidence != null ? Number(s.confidence).toFixed(2) : "—"})
                </li>
              ))}
            </ul>
          </div>
        )}

        {tradesList && tradesList.length > 0 && (
          <div>
            <div style={{ color: "var(--text-soft)", marginBottom: 4 }}>Recent trades</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Symbol</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Side</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Qty</th>
                  <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Price</th>
                </tr>
              </thead>
              <tbody>
                {tradesList.slice(0, 10).map((t, i) => (
                  <tr key={i}>
                    <td style={{ color: "var(--accent)" }}>{t.symbol ?? "—"}</td>
                    <td className="num-mono" style={{ textAlign: "right" }}>{t.side ?? "—"}</td>
                    <td className="num-mono" style={{ textAlign: "right" }}>{t.quantity ?? "—"}</td>
                    <td className="num-mono" style={{ textAlign: "right" }}>{t.price != null ? Number(t.price).toFixed(2) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {(!signalsList || signalsList.length === 0) && (!tradesList || tradesList.length === 0) && (
          <div style={{ color: "var(--text-soft)", fontSize: 11 }}>
            Run a cycle to see signals and trades here.
          </div>
        )}
      </div>
    </section>
  );
};
