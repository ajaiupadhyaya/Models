import React, { useEffect, useState } from "react";

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

export const PortfolioPanel: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [quickPredict, setQuickPredict] = useState<QuickPredict | null>(null);

  useEffect(() => {
    const fetchQuickPredict = async () => {
      try {
        const res = await fetch("/api/v1/predictions/quick-predict?symbol=AAPL");
        const json = await res.json().catch(() => ({}));
        setQuickPredict(json?.error ? { error: json.error } : json);
      } catch {
        setQuickPredict(null);
      }
    };
    fetchQuickPredict();
    const qId = setInterval(fetchQuickPredict, 60000);
    return () => clearInterval(qId);
  }, []);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        setError(null);
        const res = await fetch("/api/v1/monitoring/dashboard");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setData(json);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load");
        setData(null);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboard();
    const id = setInterval(fetchDashboard, 30000);
    return () => clearInterval(id);
  }, []);

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
      <section className="panel panel-main-secondary">
        <div className="panel-title">Portfolio & Strategies</div>
        <div className="panel-body-muted">
          Backend not reachable. Start the API on port 8000 and ensure the frontend proxy is used (npm run dev).
        </div>
      </section>
    );
  }

  const hasModels = (data?.active_models ?? 0) > 0;
  const models = data?.available_models ?? [];
  const recent = data?.recent_predictions ?? [];
  const totalPreds = data?.system?.total_predictions ?? 0;

  return (
    <section className="panel panel-main-secondary">
      <div className="panel-title">Portfolio & Strategies</div>
      <div className="panel-body-muted" style={{ fontSize: "12px" }}>
        {quickPredict && !quickPredict.error && (
          <p style={{ marginBottom: 8 }}>
            <strong>ML Signal (AAPL):</strong> {quickPredict.recommendation ?? "—"} —
            signal {(quickPredict.signal ?? 0).toFixed(2)}
            {quickPredict.current_price != null && ` @ $${quickPredict.current_price.toFixed(2)}`}
          </p>
        )}
        {!hasModels ? (
          <p>
            No models loaded yet. Train or load models via the API (<code>/api/v1/models</code>), then run backtests
            at <code>/api/v1/backtest/run</code>. Dashboard data refreshes every 30s.
          </p>
        ) : (
          <>
            <p>
              <strong>Active models:</strong> {data?.active_models ?? 0}
            </p>
            {models.length > 0 && (
              <p>
                <strong>Available:</strong> {models.join(", ")}
              </p>
            )}
            <p>
              <strong>Total predictions:</strong> {totalPreds}
            </p>
            {recent.length > 0 && (
              <>
                <strong>Recent predictions:</strong>
                <ul style={{ marginTop: 4, paddingLeft: 16 }}>
                  {recent.slice(-5).map((p, i) => (
                    <li key={i}>
                      {p.model_name} / {p.symbol}: signal {typeof p.signal === "number" ? p.signal.toFixed(2) : "—"}
                    </li>
                  ))}
                </ul>
              </>
            )}
            <p style={{ marginTop: 8, color: "var(--text-soft)" }}>
              Run backtests via API: <code>POST /api/v1/backtest/run</code>
            </p>
          </>
        )}
      </div>
    </section>
  );
};
