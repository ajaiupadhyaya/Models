/**
 * AI Insights Tab – sentiment, price targets, and AI analysis for the primary symbol.
 * Context: Real-time sentiment, predictive price targets with confidence, anomaly alerts, NL query.
 */
import React, { useEffect, useState, useCallback } from "react";
import { useFetchWithRetry, getAuthHeaders } from "../../hooks/useFetchWithRetry";
import { resolveApiUrl } from "../../apiBase";
import { useTerminal } from "../TerminalContext";
import { PanelErrorState } from "./PanelErrorState";

interface StockAnalysisResponse {
  symbol?: string;
  current_price?: number;
  technical_analysis?: string;
  prediction?: {
    next_price?: number;
    implied_change_pct?: number;
    confidence?: number;
    confidence_interval_low?: number;
    confidence_interval_high?: number;
  };
  trading_insight?: { recommendation?: string; reasoning?: string; action?: string; risk_level?: string };
  sentiment?: { score?: number; label?: string; reasoning?: string };
  error?: string;
}

function parseStockAnalysis(json: unknown): StockAnalysisResponse | null {
  if (json && typeof json === "object" && "detail" in (json as object)) return null;
  return json as StockAnalysisResponse;
}

export const AiInsightsPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const url = `/api/v1/ai/stock-analysis/${primarySymbol}?include_prediction=true`;
  const { data, error, loading, retry } = useFetchWithRetry<StockAnalysisResponse | null>(url, {
    parse: parseStockAnalysis,
    deps: [primarySymbol],
  });

  if (loading) {
    return (
      <section className="panel panel-main">
        <div className="panel-title">AI Insights: {primarySymbol}</div>
        <div className="panel-skeleton">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className={`panel-skeleton-line ${i % 2 === 0 ? "short" : "medium"}`} />
          ))}
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <PanelErrorState
        title={`AI Insights: ${primarySymbol}`}
        error={error}
        hint="Try again or set OPENAI_API_KEY on the server for full analysis."
        onRetry={retry}
      />
    );
  }

  const price = data?.current_price;
  const pred = data?.prediction;
  const insight = data?.trading_insight;
  const sentiment = data?.sentiment;
  const analysis = data?.technical_analysis;

  return (
    <section className="panel panel-main">
      <div className="panel-title">AI Insights: {data?.symbol ?? primarySymbol}</div>
      <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
        {price != null && (
          <div style={{ marginBottom: 12 }}>
            <span style={{ color: "var(--accent)" }}>Price</span>
            <span className="num-mono" style={{ marginLeft: 8 }}>${price.toFixed(2)}</span>
          </div>
        )}
        {sentiment && (sentiment.score != null || sentiment.label) && (
          <div style={{ marginBottom: 12, padding: 8, background: "var(--bg-panel)", borderRadius: 4, border: "1px solid var(--border)" }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>Sentiment</div>
            <span className="num-mono" style={{ color: (sentiment.score ?? 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)", marginRight: 8 }}>
              {sentiment.label ?? (sentiment.score != null ? (sentiment.score >= 0 ? "Bullish" : "Bearish") : "—")}
            </span>
            {sentiment.score != null && (
              <span className="num-mono" style={{ color: "var(--text-soft)" }}>Score: {(sentiment.score * 100).toFixed(0)}</span>
            )}
            {sentiment.reasoning && (
              <div style={{ color: "var(--text-soft)", fontSize: 11, marginTop: 4 }}>{sentiment.reasoning}</div>
            )}
          </div>
        )}
        {pred && (pred.next_price != null || pred.implied_change_pct != null) && (
          <div style={{ marginBottom: 12, padding: 8, background: "var(--bg-panel)", borderRadius: 4, border: "1px solid var(--border)" }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>Price target (ML)</div>
            {pred.next_price != null && (
              <span className="num-mono" style={{ marginRight: 12 }}>Next: ${pred.next_price.toFixed(2)}</span>
            )}
            {pred.confidence != null && (
              <span className="num-mono" style={{ color: "var(--text-soft)", marginRight: 8 }}>Confidence: {(pred.confidence * 100).toFixed(0)}%</span>
            )}
            {pred.implied_change_pct != null && (
              <span className="num-mono" style={{ color: pred.implied_change_pct >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                {pred.implied_change_pct >= 0 ? "+" : ""}{pred.implied_change_pct.toFixed(2)}%
              </span>
            )}
            {(pred.confidence_interval_low != null || pred.confidence_interval_high != null) && (
              <div style={{ fontSize: 11, color: "var(--text-soft)", marginTop: 4 }}>
                Range: ${(pred.confidence_interval_low ?? pred.next_price ?? 0).toFixed(2)} – ${(pred.confidence_interval_high ?? pred.next_price ?? 0).toFixed(2)}
              </div>
            )}
          </div>
        )}
        {insight && (insight.recommendation || insight.action || insight.reasoning) && (
          <div style={{ marginBottom: 12, padding: 8, background: "var(--bg-panel)", borderRadius: 4, border: "1px solid var(--border)" }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>Trading insight</div>
            {insight.action && <span className="num-mono" style={{ marginRight: 8 }}>{insight.action}</span>}
            {insight.recommendation && <div style={{ color: "var(--text-soft)", fontSize: 11, marginTop: 4 }}>{insight.recommendation}</div>}
            {insight.reasoning && <div style={{ color: "var(--text-soft)", fontSize: 11, marginTop: 4 }}>{insight.reasoning}</div>}
          </div>
        )}
        {analysis && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ color: "var(--accent)", marginBottom: 4 }}>Technical analysis</div>
            <div style={{ color: "var(--text-soft)", fontSize: 11, whiteSpace: "pre-wrap" }}>{analysis}</div>
          </div>
        )}
        {!price && !pred && !insight && !analysis && (
          <div className="panel-body-muted">No AI insights for this symbol. Try another or ensure OPENAI_API_KEY is set.</div>
        )}
      </div>
    </section>
  );
};
