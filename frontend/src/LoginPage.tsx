import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { resolveApiUrl } from "./apiBase";

const TOKEN_KEY = "terminal_token";

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

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
          setError("Auth not configured on server. Set TERMINAL_USER, TERMINAL_PASSWORD, and AUTH_SECRET in the server environment (e.g. Render dashboard).");
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
        setError("Invalid response");
      }
    } catch (err) {
      setError("Connection failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleDemo() {
    setUsername("demo");
    setPassword("demo");
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <h1 className="login-title">Bloomberg Terminal</h1>
        <p className="login-subtitle">Sign in to continue.</p>
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
            <button type="button" className="login-demo" onClick={handleDemo}>
              Use demo (demo / demo)
            </button>
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
