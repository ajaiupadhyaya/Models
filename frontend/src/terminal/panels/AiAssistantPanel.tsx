import React, { useState, useEffect, useCallback, useRef } from "react";
import { resolveApiUrl } from "../../apiBase";
import { useTerminal } from "../TerminalContext";
import { getAuthHeaders } from "../../hooks/useFetchWithRetry";

interface ToolCall {
  tool: string;
  input: Record<string, unknown>;
  status: string;
  result?: unknown;
}

interface StructuredData {
  run_dcf?: Record<string, unknown>;
  screen_stocks?: Record<string, unknown>;
  get_company_overview?: Record<string, unknown>;
  run_backtest?: Record<string, unknown>;
  get_macro_snapshot?: Record<string, unknown>;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
  structuredData?: StructuredData;
}

/** Single-word ticker-like (1–5 uppercase letters). */
function looksLikeTicker(q: string): boolean {
  const t = q.trim().toUpperCase();
  return /^[A-Z]{1,5}$/.test(t) && t.length >= 1;
}

export const AiAssistantPanel: React.FC = () => {
  const { lastAiQuery, setAiResponse } = useTerminal();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [toolCallsActive, setToolCallsActive] = useState<ToolCall[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, toolCallsActive]);

  const sendMessage = useCallback(
    async (text: string, isCommandBar = false) => {
      const q = text.trim();
      if (!q) return;
      setLoading(true);
      if (!isCommandBar) {
        setMessages((prev) => [...prev, { role: "user", content: q }]);
        setInput("");
      }
      setToolCallsActive([]);

      try {
        const history = messages.map((m) => ({ role: m.role, content: m.content }));
        const res = await fetch(resolveApiUrl("/api/v1/ai/chat"), {
          method: "POST",
          headers: { "Content-Type": "application/json", ...getAuthHeaders() },
          body: JSON.stringify({ message: q, history }),
        });
        const json = await res.json().catch(() => ({}));

        if (!res.ok) {
          const err = json.detail ?? json.error ?? `HTTP ${res.status}`;
          if (isCommandBar) setAiResponse(err);
          else setMessages((prev) => [...prev, { role: "assistant", content: err }]);
          return;
        }

        const reply = json.reply ?? "";
        const toolCalls = (json.tool_calls ?? []) as ToolCall[];
        const structuredData = (json.structured_data ?? {}) as StructuredData;

        if (isCommandBar) {
          setAiResponse(reply);
        } else {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: reply,
              toolCalls: toolCalls.length ? toolCalls : undefined,
              structuredData: Object.keys(structuredData).length ? structuredData : undefined,
            },
          ]);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Request failed";
        if (isCommandBar) setAiResponse(msg);
        else setMessages((prev) => [...prev, { role: "assistant", content: msg }]);
      } finally {
        setLoading(false);
        setToolCallsActive([]);
      }
    },
    [messages, setAiResponse]
  );

  useEffect(() => {
    if (!lastAiQuery?.q || lastAiQuery.a !== "") return;
    const q = lastAiQuery.q.trim();
    if (looksLikeTicker(q)) {
      sendMessage(`Analyze ${q} for me`, true);
    } else {
      sendMessage(q, true);
    }
  }, [lastAiQuery?.q, lastAiQuery?.a]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="panel panel-right bg-gray-900 border border-gray-700 rounded-lg">
      <div className="panel-title text-white">AI Assistant</div>
      <div className="flex flex-col h-full min-h-0">
        <div className="flex-1 overflow-y-auto p-2 space-y-3" ref={scrollRef}>
          {messages.length === 0 && !lastAiQuery && (
            <p className="text-gray-500 text-sm">
              Ask about stocks, run DCF, screen stocks, get company overviews, run backtests, or macro snapshots. Example: &quot;Run DCF on AAPL at 10% WACC&quot;
            </p>
          )}
          {lastAiQuery != null && messages.length === 0 && (
            <div className="space-y-2">
              <div>
                <span className="text-blue-400 font-mono">Q: </span>
                <span className="text-white">{lastAiQuery.q}</span>
              </div>
              <div>
                <span className="text-green-400 font-mono">A: </span>
                <span className="text-gray-300">
                  {loading ? "Thinking..." : lastAiQuery.a || "—"}
                </span>
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className="space-y-1">
              <div>
                <span className={`font-mono ${m.role === "user" ? "text-blue-400" : "text-green-400"}`}>
                  {m.role === "user" ? "Q: " : "A: "}
                </span>
                <span className="text-white whitespace-pre-wrap">{m.content}</span>
              </div>
              {m.toolCalls && m.toolCalls.length > 0 && (
                <div className="ml-4 space-y-1 text-xs">
                  {m.toolCalls.map((tc, j) => (
                    <div key={j} className="text-gray-400 flex items-center gap-2">
                      <span>🔧</span>
                      <span>
                        {tc.tool === "run_dcf" && tc.input?.symbol && `Running DCF on ${tc.input.symbol}...`}
                        {tc.tool === "screen_stocks" && "Running screener..."}
                        {tc.tool === "get_company_overview" && tc.input?.symbol && `Getting overview for ${tc.input.symbol}...`}
                        {tc.tool === "run_backtest" && "Running backtest..."}
                        {tc.tool === "get_macro_snapshot" && "Getting macro snapshot..."}
                        {!["run_dcf", "screen_stocks", "get_company_overview", "run_backtest", "get_macro_snapshot"].includes(tc.tool) && `${tc.tool}`}
                      </span>
                      {tc.status === "complete" && <span className="text-green-500">✓</span>}
                    </div>
                  ))}
                </div>
              )}
              {m.structuredData && Object.keys(m.structuredData).length > 0 && (
                <div className="ml-4 mt-2 space-y-2">
                  {m.structuredData.run_dcf && !("error" in m.structuredData.run_dcf) && (
                    <div className="bg-gray-800 border border-gray-600 rounded p-2 text-xs">
                      <div className="text-blue-400 font-semibold mb-1">DCF Result</div>
                      <div className="text-gray-300">
                        Intrinsic: ${(m.structuredData.run_dcf as Record<string, unknown>).intrinsic_value_per_share}
                        {" | "}
                        Current: ${(m.structuredData.run_dcf as Record<string, unknown>).current_price}
                        {" | "}
                        Upside: {(m.structuredData.run_dcf as Record<string, unknown>).upside_downside_pct}%
                      </div>
                    </div>
                  )}
                  {m.structuredData.screen_stocks && !("error" in m.structuredData.screen_stocks) && (
                    <div className="bg-gray-800 border border-gray-600 rounded p-2 text-xs">
                      <div className="text-blue-400 font-semibold mb-1">Screener Results (top {(m.structuredData.screen_stocks as Record<string, unknown>).count})</div>
                      <div className="text-gray-300 space-y-0.5">
                        {((m.structuredData.screen_stocks as Record<string, unknown>).tickers as Array<Record<string, unknown>>)?.slice(0, 5).map((t, k) => (
                          <div key={k}>
                            {t.symbol}: {t.name} | MCap: {t.market_cap != null ? `$${Number(t.market_cap) / 1e9}B` : "—"} | P/E: {t.pe ?? "—"}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {m.structuredData.get_company_overview && !("error" in m.structuredData.get_company_overview) && (
                    <div className="bg-gray-800 border border-gray-600 rounded p-2 text-xs">
                      <div className="text-blue-400 font-semibold mb-1">Company Overview</div>
                      <div className="text-gray-300">
                        {(m.structuredData.get_company_overview as Record<string, unknown>).name} | Price: ${(m.structuredData.get_company_overview as Record<string, unknown>).price} | Sector: {(m.structuredData.get_company_overview as Record<string, unknown>).sector}
                      </div>
                    </div>
                  )}
                  {m.structuredData.run_backtest && !("error" in m.structuredData.run_backtest) && (
                    <div className="bg-gray-800 border border-gray-600 rounded p-2 text-xs">
                      <div className="text-blue-400 font-semibold mb-1">Backtest Result</div>
                      <div className="text-gray-300">
                        Sharpe: {(m.structuredData.run_backtest as Record<string, unknown>).sharpe_ratio} | CAGR: {(m.structuredData.run_backtest as Record<string, unknown>).cagr_pct}% | Max DD: {(m.structuredData.run_backtest as Record<string, unknown>).max_drawdown_pct}%
                      </div>
                    </div>
                  )}
                  {m.structuredData.get_macro_snapshot && !("error" in m.structuredData.get_macro_snapshot) && (
                    <div className="bg-gray-800 border border-gray-600 rounded p-2 text-xs">
                      <div className="text-blue-400 font-semibold mb-1">Macro Snapshot</div>
                      <div className="text-gray-300">
                        {JSON.stringify(m.structuredData.get_macro_snapshot)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {loading && toolCallsActive.length === 0 && messages.length > 0 && (
            <div className="text-gray-500 text-sm">Thinking...</div>
          )}
        </div>
        <div className="p-2 border-t border-gray-700">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendMessage(input, false);
            }}
            className="flex gap-2"
          >
            <input
              className="flex-1 bg-gray-800 border border-gray-600 text-white px-3 py-2 rounded text-sm"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about stocks, DCF, screener..."
              disabled={loading}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm disabled:opacity-50"
              disabled={loading}
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </section>
  );
};
