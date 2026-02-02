import React from "react";
import { useFetchWithRetry } from "../../hooks/useFetchWithRetry";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface NewsItem {
  title?: string;
  summary?: string;
  url?: string;
  published?: string;
  sentiment?: number | string;
}

interface MarketSummaryResponse {
  detail?: unknown;
  analyses?: Record<string, { analysis?: string }>;
}

function parseNews(json: unknown): NewsItem[] | null {
  const r = json as MarketSummaryResponse;
  if (r?.detail) return null;
  if (r?.analyses && typeof r.analyses === "object") {
    return Object.entries(r.analyses).map(([sym, v]) => ({
      title: v?.analysis ?? `Summary for ${sym}`,
      published: new Date().toISOString().slice(0, 10),
    }));
  }
  return [];
}

export const NewsPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const url = `/api/v1/ai/market-summary?symbols=${primarySymbol}`;
  const { data, error, loading, retry } = useFetchWithRetry<NewsItem[] | null>(url, {
    parse: parseNews,
    deps: [primarySymbol],
  });

  const items = data ?? [];

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
        hint="News/sentiment API may require configuration."
        onRetry={retry}
      />
    );
  }

  return (
    <section className="panel panel-main">
      <div className="panel-title">News: {primarySymbol}</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        {items.length > 0 ? (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {items.slice(0, 10).map((item, i) => (
              <li key={i} style={{ marginBottom: 8 }}>
                <span style={{ color: "var(--text)" }}>{item.title ?? item.summary ?? "—"}</span>
                {item.published && (
                  <span style={{ color: "var(--text-soft)", marginLeft: 8 }}>{item.published}</span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <div className="panel-body-muted">
            No headlines. Use sentiment or news API for live feed.
          </div>
        )}
      </div>
    </section>
  );
};
