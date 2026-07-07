import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { resolveApiUrl } from "./apiBase";

const TOKEN_KEY = "terminal_token";

const FEATURES = [
  "Candlestick charts with technical indicators",
  "Factor exposure, regime detection, and quant models",
  "Portfolio risk metrics and strategy backtesting",
  "AI research assistant with natural-language queries",
];

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [authConfigured, setAuthConfigured] = useState<boolean | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetch(resolveApiUrl("/health"))
      .then((r) => setApiHealthy(r.ok))
      .catch(() => setApiHealthy(false));
    fetch(resolveApiUrl("/api/auth/status"))
      .then((r) => r.json().catch(() => ({ configured: true })))
      .then((data) => setAuthConfigured(data?.configured === true))
      .catch(() => setAuthConfigured(true));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(resolveApiUrl("/api/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const detail = typeof data.detail === "string" ? data.detail : "";
        if (res.status === 503 && /auth|configured/i.test(detail)) {
          setError(
            "Auth is not configured on this server. Set TERMINAL_USER, TERMINAL_PASSWORD, and AUTH_SECRET in the deployment environment."
          );
        } else {
          setError(detail || "Invalid username or password");
        }
        return;
      }
      const data = await res.json();
      if (data.token) {
        localStorage.setItem(TOKEN_KEY, data.token);
        navigate("/", { replace: true });
      } else {
        setError("Invalid response from server");
      }
    } catch {
      setError("Cannot reach the API. Start the backend or check your deployment URL.");
    } finally {
      setLoading(false);
    }
  }

  function handleDemo() {
    setUsername("demo");
    setPassword("demo");
  }

  const showDemoButton = import.meta.env.DEV;

  return (
    <div className="login-page">
      <div className="login-card">
        <h1 className="login-title">Models Quant Terminal</h1>
        <p className="login-subtitle">
          Bloomberg-style research terminal for charts, quant models, backtesting, and AI commentary.
        </p>

        <ul className="login-features">
          {FEATURES.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>

        {apiHealthy === false && (
          <p className="login-notice login-notice-warn">
            API unreachable. For local use, run <code>uvicorn api.main:app --reload --port 8000</code> then refresh.
          </p>
        )}

        {authConfigured === false && (
          <p className="login-notice">
            Auth is not configured — local dev can open the terminal without signing in. See{" "}
            <a href="https://github.com/ajaiupadhyaya/Models/blob/main/GETTING_STARTED.md" target="_blank" rel="noreferrer">
              GETTING_STARTED.md
            </a>
            .
          </p>
        )}

        {authConfigured === true && !showDemoButton && (
          <p className="login-notice">
            Sign in with credentials from your deployment secrets. No public demo password is published.
          </p>
        )}

        <form onSubmit={handleSubmit} className="login-form">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
          {error && <p className="login-error">{error}</p>}
          <div className="login-actions">
            <button type="submit" disabled={loading} className={loading ? "login-loading" : ""}>
              {loading ? (
                <>
                  <span className="login-spinner" aria-hidden />
                  Signing in…
                </>
              ) : (
                "Sign in"
              )}
            </button>
            {showDemoButton && (
              <button type="button" className="login-demo" onClick={handleDemo}>
                Fill demo credentials (local dev)
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}
