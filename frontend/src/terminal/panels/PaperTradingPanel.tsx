import React, { useState, useCallback, useEffect } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface HealthData {
  status?: string;
  configured?: boolean;
  connected?: boolean;
  error?: string;
}

interface PositionItem {
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

interface PositionsResponse {
  detail?: unknown;
  positions?: Record<string, PositionItem>;
}

interface PortfolioPosition {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

interface PortfolioData {
  detail?: unknown;
  cash?: number;
  portfolio_value?: number;
  total_value?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
  positions?: PortfolioPosition[];
}

function parseHealth(json: unknown): HealthData | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as HealthData;
}

function parsePositions(json: unknown): Record<string, PositionItem> | null {
  const r = json as PositionsResponse;
  if (r?.detail) return null;
  return r?.positions ?? null;
}

function parsePortfolio(json: unknown): PortfolioData | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as PortfolioData;
}

export const PaperTradingPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const [orderSymbol, setOrderSymbol] = useState(primarySymbol);
  const [orderSide, setOrderSide] = useState<"buy" | "sell">("buy");
  const [orderQty, setOrderQty] = useState("");
  const [orderSubmitting, setOrderSubmitting] = useState(false);
  const [orderMessage, setOrderMessage] = useState<string | null>(null);
  const [executeSignalLoading, setExecuteSignalLoading] = useState(false);
  const [executeSignalMessage, setExecuteSignalMessage] = useState<string | null>(null);

  const { data: healthData, error: healthError, loading: healthLoading, retry: healthRetry } = useFetchWithRetry<HealthData | null>(
    "/api/v1/paper-trading/health",
    { parse: parseHealth }
  );
  const configured = healthData?.configured ?? false;
  const connected = healthData?.connected ?? false;

  const positionsUrl = configured ? "/api/v1/paper-trading/positions" : null;
  const { data: positionsData, error: positionsError, loading: positionsLoading, retry: positionsRetry } = useFetchWithRetry<Record<string, PositionItem> | null>(
    positionsUrl,
    { parse: parsePositions, deps: [configured] }
  );
  const portfolioUrl = configured ? "/api/v1/paper-trading/portfolio" : null;
  const { data: portfolioData, error: portfolioError, loading: portfolioLoading, retry: portfolioRetry } = useFetchWithRetry<PortfolioData | null>(
    portfolioUrl,
    { parse: parsePortfolio, deps: [configured] }
  );

  useEffect(() => setOrderSymbol(primarySymbol), [primarySymbol]);

  const placeOrder = useCallback(async () => {
    const qty = parseFloat(orderQty);
    if (!Number.isFinite(qty) || qty <= 0) {
      setOrderMessage("Enter a valid quantity.");
      return;
    }
    setOrderSubmitting(true);
    setOrderMessage(null);
    try {
      const res = await fetch(resolveApiUrl("/api/v1/paper-trading/orders/place"), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          symbol: orderSymbol,
          quantity: qty,
          side: orderSide,
          order_type: "market",
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setOrderMessage((json?.detail ?? `HTTP ${res.status}`) as string);
        return;
      }
      setOrderMessage(`Order ${json?.status ?? "submitted"}. ${json?.message ?? ""}`);
      positionsRetry();
      portfolioRetry();
    } catch (err) {
      setOrderMessage(err instanceof Error ? err.message : "Request failed");
    } finally {
      setOrderSubmitting(false);
    }
  }, [orderSymbol, orderSide, orderQty, positionsRetry, portfolioRetry]);

  const runExecuteSignal = useCallback(async () => {
    setExecuteSignalLoading(true);
    setExecuteSignalMessage(null);
    try {
      const predRes = await fetch(resolveApiUrl(`/api/v1/predictions/quick-predict?symbol=${primarySymbol}`), { headers: getAuthHeaders() });
      const predJson = await predRes.json().catch(() => ({}));
      const signal = predJson?.signal ?? 0;
      const confidence = Math.min(1, Math.abs(signal));
      const priceRes = await fetch(resolveApiUrl(`/api/v1/backtest/sample-data?symbol=${primarySymbol}&period=1d`), { headers: getAuthHeaders() });
      const priceJson = await priceRes.json().catch(() => ({}));
      const candles = priceJson?.candles ?? [];
      const currentPrice = candles.length > 0 ? (candles[candles.length - 1] as { close?: number }).close : 0;
      const res = await fetch(resolveApiUrl("/api/v1/paper-trading/execute-signal"), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          symbol: primarySymbol,
          signal,
          confidence,
          current_price: currentPrice || 0,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setExecuteSignalMessage((json?.detail ?? `HTTP ${res.status}`) as string);
        return;
      }
      setExecuteSignalMessage(`Signal: ${json?.status ?? "hold"}. ${json?.order_id ? `Order ${json.order_id}` : ""}`);
      positionsRetry();
      portfolioRetry();
    } catch (err) {
      setExecuteSignalMessage(err instanceof Error ? err.message : "Request failed");
    } finally {
      setExecuteSignalLoading(false);
    }
  }, [primarySymbol, positionsRetry, portfolioRetry]);

  if (healthLoading && !healthData) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Paper Trading</div>
        <div className="panel-body-muted">Loading…</div>
      </section>
    );
  }

  if (healthError && !healthData) {
    return (
      <PanelErrorState
        title="Paper Trading"
        error={healthError}
        hint="Paper trading requires ALPACA_API_KEY and ALPACA_API_SECRET on the server."
        onRetry={healthRetry}
      />
    );
  }

  if (!configured) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">Paper Trading</div>
        <div className="panel-error-box" style={{ marginTop: 8 }}>
          <div style={{ color: "var(--accent-red)", marginBottom: 4 }}>Not configured</div>
          <div style={{ fontSize: 11, color: "var(--text-soft)" }}>
            Set ALPACA_API_KEY and ALPACA_API_SECRET in .env (and optionally ALPACA_API_BASE for paper). See LAUNCH_GUIDE.md.
          </div>
        </div>
      </section>
    );
  }

  const positionsList: Array<{ symbol: string; quantity: number; entry_price: number; current_price: number; unrealized_pnl_pct: number }> =
    (portfolioData?.positions && Array.isArray(portfolioData.positions))
      ? portfolioData.positions
      : (positionsData
          ? Object.entries(positionsData).map(([sym, p]) => ({
              symbol: sym,
              quantity: p.quantity,
              entry_price: p.entry_price,
              current_price: p.current_price,
              unrealized_pnl_pct: p.unrealized_pnl_pct,
            }))
          : []);

  return (
    <section className="panel panel-main">
      <div className="panel-title">Paper Trading</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12 }}>
          <span style={{ color: "var(--text-soft)" }}>Status: </span>
          <span style={{ color: connected ? "var(--accent-green)" : "var(--accent-red)" }}>
            {connected ? "Connected" : "Not connected"}
          </span>
          <button type="button" className="ai-button" style={{ marginLeft: 8 }} onClick={() => { healthRetry(); positionsRetry(); portfolioRetry(); }}>
            Refresh
          </button>
        </div>

        {(portfolioLoading && !portfolioData) ? (
          <div className="panel-body-muted">Loading portfolio…</div>
        ) : portfolioError ? (
          <div style={{ color: "var(--accent-red)", marginBottom: 8 }}>{portfolioError}</div>
        ) : portfolioData ? (
          <>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, marginBottom: 12 }}>
              <tbody>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Cash</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>${Number(portfolioData.cash ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Portfolio value</td>
                  <td className="num-mono" style={{ textAlign: "right" }}>${Number(portfolioData.portfolio_value ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-soft)", padding: "2px 8px 2px 0" }}>Total PnL</td>
                  <td className="num-mono" style={{ textAlign: "right", color: (portfolioData.total_pnl ?? 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                    ${Number(portfolioData.total_pnl ?? 0).toFixed(2)} ({Number(portfolioData.total_pnl_pct ?? 0).toFixed(2)}%)
                  </td>
                </tr>
              </tbody>
            </table>

            {positionsList.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ color: "var(--text-soft)", marginBottom: 4 }}>Positions</div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", color: "var(--text-soft)", fontWeight: 500 }}>Symbol</th>
                      <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Qty</th>
                      <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Entry</th>
                      <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>Price</th>
                      <th style={{ textAlign: "right", color: "var(--text-soft)", fontWeight: 500 }}>PnL %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positionsList.map((p) => (
                      <tr key={p.symbol}>
                        <td style={{ color: "var(--accent)" }}>{p.symbol}</td>
                        <td className="num-mono" style={{ textAlign: "right" }}>{p.quantity}</td>
                        <td className="num-mono" style={{ textAlign: "right" }}>${Number(p.entry_price).toFixed(2)}</td>
                        <td className="num-mono" style={{ textAlign: "right" }}>${Number(p.current_price).toFixed(2)}</td>
                        <td className="num-mono" style={{ textAlign: "right", color: p.unrealized_pnl_pct >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {Number(p.unrealized_pnl_pct).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        ) : null}

        <div style={{ marginBottom: 8, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 6 }}>Place order</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 6 }}>
            <input
              type="text"
              className="ai-input"
              placeholder="Symbol"
              value={orderSymbol}
              onChange={(e) => setOrderSymbol(e.target.value.toUpperCase())}
              style={{ width: 72 }}
              aria-label="Order symbol"
            />
            <select
              value={orderSide}
              onChange={(e) => setOrderSide(e.target.value as "buy" | "sell")}
              style={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                padding: "6px 8px",
                borderRadius: 4,
                fontSize: 12,
              }}
              aria-label="Side"
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
            <input
              type="number"
              className="ai-input"
              placeholder="Quantity"
              value={orderQty}
              onChange={(e) => setOrderQty(e.target.value)}
              min="0"
              step="0.01"
              style={{ width: 80 }}
              aria-label="Quantity"
            />
            <button type="button" className="ai-button" disabled={orderSubmitting} onClick={placeOrder}>
              {orderSubmitting ? "Submitting…" : "Place order"}
            </button>
          </div>
          {orderMessage && <div style={{ fontSize: 11, color: "var(--text-soft)" }}>{orderMessage}</div>}
        </div>

        <div style={{ paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ color: "var(--text-soft)", marginBottom: 6 }}>Execute signal ({primarySymbol})</div>
          <button type="button" className="ai-button" disabled={executeSignalLoading} onClick={runExecuteSignal}>
            {executeSignalLoading ? "Running…" : "Execute ML signal"}
          </button>
          {executeSignalMessage && <div style={{ fontSize: 11, color: "var(--text-soft)", marginTop: 4 }}>{executeSignalMessage}</div>}
        </div>
      </div>
    </section>
  );
};
