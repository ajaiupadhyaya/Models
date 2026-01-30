import React, { useState } from "react";

export const AiAssistantPanel: React.FC = () => {
  const [symbol, setSymbol] = useState("AAPL");
  const [response, setResponse] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const runAnalysis = async () => {
    try {
      setLoading(true);
      setResponse("");
      const res = await fetch(`/api/v1/ai/stock-analysis/${symbol}?include_prediction=true`);
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        setResponse(`Error ${res.status}: ${json?.detail ?? res.statusText ?? "AI analysis failed"}. Ensure API is running and OPENAI_API_KEY is set.`);
        return;
      }
      const analysis = json.technical_analysis ?? "";
      const insight = json.trading_insight ?? {};
      const trading = insight.reasoning ?? insight.recommendation ?? "";
      setResponse([analysis, trading ? `Trading: ${trading}` : ""].filter(Boolean).join("\n\n"));
    } catch (err) {
      setResponse("Unable to reach AI analysis endpoint. Start the API on port 8000 and use npm run dev so /api is proxied.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel panel-right">
      <div className="panel-title">AI Assistant</div>
      <div className="ai-controls">
        <input
          className="ai-input"
          value={symbol}
          onChange={e => setSymbol(e.target.value.toUpperCase())}
          placeholder="Symbol"
        />
        <button className="ai-button" onClick={runAnalysis} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      <pre className="ai-output">{response || "Ask the assistant about any symbol."}</pre>
    </section>
  );
};

