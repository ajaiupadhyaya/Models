import React, { useState, useEffect, useCallback } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useTerminal } from "../TerminalContext";
import { getAuthHeaders } from "../../hooks/useFetchWithRetry";

/** Single-word ticker-like (1–5 uppercase letters). */
function looksLikeTicker(q: string): boolean {
  const t = q.trim().toUpperCase();
  return /^[A-Z]{1,5}$/.test(t) && t.length >= 1;
}

export const AiAssistantPanel: React.FC = () => {
  const { primarySymbol, lastAiQuery, setLastAiQuery, setAiResponse } = useTerminal();
  const [manualResponse, setManualResponse] = useState<string>("");
  const [symbol, setSymbol] = useState(primarySymbol);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSymbol(primarySymbol);
  }, [primarySymbol]);

  const runAnalysis = useCallback(
    async (sym: string, isCommandBarQuery?: boolean) => {
      try {
        setLoading(true);
        if (!isCommandBarQuery) setManualResponse("");
        const res = await fetch(resolveApiUrl(`/api/v1/ai/stock-analysis/${sym}?include_prediction=true`), { headers: getAuthHeaders() });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) {
          const msg = `Error ${res.status}: ${json?.detail ?? res.statusText ?? "AI analysis failed"}. Ensure OPENAI_API_KEY is set on the server for full analysis.`;
          if (isCommandBarQuery) setAiResponse(msg);
          else setManualResponse(msg);
          return;
        }
        const analysis = json.technical_analysis ?? "";
        const insight = json.trading_insight ?? {};
        const trading = insight.reasoning ?? insight.recommendation ?? "";
        const out = [analysis, trading ? `Trading: ${trading}` : ""].filter(Boolean).join("\n\n");
        if (isCommandBarQuery) setAiResponse(out);
        else setManualResponse(out);
      } catch (err) {
        const msg = "AI analysis unavailable. Please try again.";
        if (isCommandBarQuery) setAiResponse(msg);
        else setManualResponse(msg);
      } finally {
        setLoading(false);
      }
    },
    [setAiResponse]
  );

  const runNlQuery = useCallback(
    async (question: string, isCommandBarQuery?: boolean) => {
      try {
        setLoading(true);
        if (!isCommandBarQuery) setManualResponse("");
        const res = await fetch(
          resolveApiUrl(`/api/v1/ai/nl-query?q=${encodeURIComponent(question)}`),
          { headers: getAuthHeaders() }
        );
        const json = await res.json().catch(() => ({}));
        if (!res.ok) {
          const msg = `Error ${res.status}: ${json?.detail ?? res.statusText ?? "NL query failed"}.`;
          if (isCommandBarQuery) setAiResponse(msg);
          else setManualResponse(msg);
          return;
        }
        const answer = json.answer ?? "No answer returned.";
        if (isCommandBarQuery) setAiResponse(answer);
        else setManualResponse(answer);
      } catch (err) {
        const msg = "Query unavailable. Please try again.";
        if (isCommandBarQuery) setAiResponse(msg);
        else setManualResponse(msg);
      } finally {
        setLoading(false);
      }
    },
    [setAiResponse]
  );

  useEffect(() => {
    if (!lastAiQuery?.q || lastAiQuery.a !== "") return;
    const q = lastAiQuery.q.trim();
    if (looksLikeTicker(q)) {
      runAnalysis(q, true);
    } else {
      runNlQuery(q, true);
    }
  }, [lastAiQuery?.q, lastAiQuery?.a, runAnalysis, runNlQuery]);

  const handleManualAnalyze = () => {
    setLastAiQuery(null);
    runAnalysis(symbol, false);
  };

  return (
    <section className="panel panel-right">
      <div className="panel-title">AI Assistant</div>
      <div className="ai-controls">
        <input
          className="ai-input"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          placeholder="Symbol"
        />
        <button className="ai-button" onClick={handleManualAnalyze} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      <pre className="ai-output">
        {lastAiQuery != null ? (
          <>
            <span style={{ color: "var(--accent)" }}>Q: </span>
            {lastAiQuery.q}
            {"\n\n"}
            <span style={{ color: "var(--accent-green)" }}>A: </span>
            {loading && lastAiQuery.a === "" ? "Analyzing…" : lastAiQuery.a || "—"}
          </>
        ) : (
          manualResponse || "Ask the assistant about any symbol. Or use the command bar (e.g. type a question and Enter)."
        )}
      </pre>
    </section>
  );
};
