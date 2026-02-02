import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface NewsItem {
  title?: string;
  summary?: string;
  url?: string;
  published?: string;
  source?: string;
}

interface NewsResponse {
  detail?: unknown;
  items?: NewsItem[];
  error?: string;
  symbol?: string;
}

function parseNewsResponse(json: unknown): { items: NewsItem[]; error?: string } {
  const r = json as NewsResponse;
  if (r?.detail) return { items: [] };
  const items = Array.isArray(r?.items) ? r.items : [];
  const error = typeof r?.error === "string" ? r.error : undefined;
  return { items, error };
}

export const NewsPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const url = `/api/v1/data/news?symbol=${encodeURIComponent(primarySymbol)}&limit=15`;
  const { data, error, loading, retry } = useFetchWithRetry<{ items: NewsItem[]; error?: string } | null>(url, {
    parse: parseNewsResponse,
    deps: [primarySymbol],
  });

  const parsed = data ?? { items: [] };
  const items = parsed.items ?? [];
  const configError = parsed.error;

  if (loading) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">News: {primarySymbol}</div>
        <div className="panel-body-muted">Loading…</div>
      </section>
    );
  }

  if (error) {
    return (
      <PanelErrorState
        title={`News: ${primarySymbol}`}
        error={error}
        hint="Ensure API is running. For real headlines set FINNHUB_API_KEY (see .env.example)."
        onRetry={retry}
      />
    );
  }

  return (
    <section className="panel panel-main">
      <div className="panel-title">News: {primarySymbol}</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        {configError && (
          <div className="panel-error-inline" style={{ marginBottom: 8, color: "var(--text-soft)", fontSize: 11 }}>
            {configError}
          </div>
        )}
        {items.length > 0 ? (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {items.slice(0, 15).map((item, i) => (
              <li key={i} style={{ marginBottom: 8 }}>
                {item.url ? (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "var(--accent)", textDecoration: "none" }}
                  >
                    {item.title ?? item.summary ?? "—"}
                  </a>
                ) : (
                  <span style={{ color: "var(--text)" }}>{item.title ?? item.summary ?? "—"}</span>
                )}
                {item.published && (
                  <span style={{ color: "var(--text-soft)", marginLeft: 8 }}>
                    {item.published.slice(0, 10)}
                  </span>
                )}
                {item.source && (
                  <span style={{ color: "var(--text-soft)", marginLeft: 4 }}>({item.source})</span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <div className="panel-body-muted">
            No headlines. Set FINNHUB_API_KEY in .env for real news (see .env.example).
          </div>
        )}
      </div>
    </section>
  );
};
