import React, { useState, useCallback, useEffect } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface NewsItem {
  title?: string;
  summary?: string;
  url?: string;
  published?: string;
  source?: string;
  sentiment_score?: number;
}

interface NewsSentimentResponse {
  symbol?: string;
  items?: NewsItem[];
  aggregate_sentiment_7d?: number;
  article_count?: number;
  error?: string;
}

function sentimentGauge(value: number): { label: string; color: string } {
  if (value >= 0.2) return { label: "Bullish", color: "var(--accent-green)" };
  if (value <= -0.2) return { label: "Bearish", color: "var(--accent-red)" };
  return { label: "Neutral", color: "var(--text-soft)" };
}

export const NewsSentimentPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const [symbol, setSymbol] = useState(primarySymbol);
  const [data, setData] = useState<NewsSentimentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchNews = useCallback(async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch(resolveApiUrl(`/api/v1/news/${encodeURIComponent(symbol.toUpperCase())}?limit=20&days=7`));
      const json = (await res.json().catch(() => ({}))) as NewsSentimentResponse;
      if (!res.ok) {
        setError((json as { detail?: string }).detail ?? json.error ?? `HTTP ${res.status}`);
        return;
      }
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    if (symbol) fetchNews();
  }, [symbol]); // eslint-disable-line react-hooks/exhaustive-deps

  const agg = data?.aggregate_sentiment_7d ?? 0;
  const gauge = sentimentGauge(agg);

  return (
    <section className="panel panel-main">
      <div className="panel-title">News & Sentiment</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        <div style={{ marginBottom: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <input
            type="text"
            className="ai-input"
            placeholder="Symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            style={{ width: 80 }}
          />
          <button type="button" className="ai-button" disabled={loading} onClick={fetchNews}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {error && (
          <PanelErrorState title="News & Sentiment" error={error} hint="Set FINNHUB_API_KEY or NEWSAPI_KEY in .env for real news." onRetry={fetchNews} sectionClassName="panel panel-main-secondary" />
        )}

        {!error && loading && !data?.items?.length && (
          <div style={{ color: "var(--text-soft)", padding: 24, textAlign: "center" }}>
            Loading news and computing sentiment…
          </div>
        )}

        {!error && !loading && data && (
          <>
            {data.aggregate_sentiment_7d != null && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ color: "var(--text-soft)", marginBottom: 6, fontWeight: 600 }}>7-day aggregate sentiment</div>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div
                    title={`Score: ${agg.toFixed(2)} (${gauge.label})`}
                    style={{
                      width: 120,
                      height: 24,
                      background: "linear-gradient(to right, var(--accent-red) 0%, var(--text-soft) 50%, var(--accent-green) 100%)",
                      borderRadius: 4,
                      position: "relative",
                      overflow: "hidden",
                      cursor: "help",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        left: `${Math.max(0, Math.min(100, (agg + 1) * 50))}%`,
                        top: 0,
                        bottom: 0,
                        width: 2,
                        background: "var(--text)",
                      }}
                    />
                  </div>
                  <span style={{ color: gauge.color, fontWeight: 600 }}>{gauge.label}</span>
                  <span className="num-mono" style={{ color: "var(--text-soft)" }}>({agg.toFixed(2)})</span>
                </div>
              </div>
            )}

            {data.items && data.items.length > 0 ? (
              <div>
                <div style={{ color: "var(--text-soft)", marginBottom: 8, fontWeight: 600 }}>Articles (VADER sentiment)</div>
                <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                  {data.items.map((item, i) => (
                    <li key={i} style={{ marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
                      {item.url ? (
                        <a href={item.url} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)", textDecoration: "none" }}>
                          {item.title ?? item.summary ?? "—"}
                        </a>
                      ) : (
                        <span style={{ color: "var(--text)" }}>{item.title ?? item.summary ?? "—"}</span>
                      )}
                      {item.published && <span style={{ color: "var(--text-soft)", marginLeft: 8, fontSize: 10 }}>{item.published.slice(0, 10)}</span>}
                      {item.sentiment_score != null && (
                        <span
                          style={{
                            marginLeft: 8,
                            fontSize: 10,
                            color: item.sentiment_score >= 0.2 ? "var(--accent-green)" : item.sentiment_score <= -0.2 ? "var(--accent-red)" : "var(--text-soft)",
                          }}
                        >
                          ({item.sentiment_score.toFixed(2)})
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ) : data.error ? (
              <div style={{ color: "var(--text-soft)", padding: 16 }}>
                {data.error}
              </div>
            ) : (
              <div style={{ color: "var(--text-soft)", padding: 24, textAlign: "center" }}>
                No articles found. Configure FINNHUB_API_KEY or NEWSAPI_KEY in .env.
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
};
