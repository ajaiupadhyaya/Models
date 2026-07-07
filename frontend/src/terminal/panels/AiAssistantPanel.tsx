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

function looksLikeTicker(q: string): boolean {
  const t = q.trim().toUpperCase();
  return /^[A-Z]{1,5}$/.test(t) && t.length >= 1;
}

function formatAiError(raw: string, status?: number): string {
  if (status === 401) {
    return "Sign in required for AI features. Configure TERMINAL_USER/PASSWORD/AUTH_SECRET or use local dev without auth.";
  }
  if (/not configured|OPENAI|ANTHROPIC|API_KEY/i.test(raw)) {
    return "AI provider not configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in the server environment to enable live answers. Other terminal panels still work without AI.";
  }
  return raw;
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
          const raw = String(json.detail ?? json.error ?? `HTTP ${res.status}`);
          const err = formatAiError(raw, res.status);
          if (isCommandBar) setAiResponse(err);
          else setMessages((prev) => [...prev, { role: "assistant", content: err }]);
          return;
        }

        const reply = json.reply ?? "";
        const toolCalls = (json.tool_calls ?? []) as ToolCall[];
        const structuredData = (json.structured_data ?? {}) as StructuredData;

        if (isCommandBar) {
          setAiResponse(reply || "No response returned.");
        } else {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: reply || "No response returned.",
              toolCalls: toolCalls.length ? toolCalls : undefined,
              structuredData: Object.keys(structuredData).length ? structuredData : undefined,
            },
          ]);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Request failed";
        const friendly = formatAiError(msg);
        if (isCommandBar) setAiResponse(friendly);
        else setMessages((prev) => [...prev, { role: "assistant", content: friendly }]);
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
    <div className="flex flex-col h-full bg-surface-container-low text-on-surface">
      <div className="font-label-xs text-label-xs uppercase text-on-tertiary-container tracking-[0.4em] mb-4">AI ASSISTANT</div>
      <p className="text-tertiary font-data-mono text-[10px] mb-4 uppercase">
        Requires an LLM API key on the server. Without it, you still get charts, quant, portfolio, and backtests.
      </p>
      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex-1 overflow-y-auto mb-4 font-data-mono text-[12px] space-y-4" ref={scrollRef}>
          {messages.length === 0 && !lastAiQuery && (
            <p className="text-on-surface-variant uppercase">
              Ask about stocks, run DCF, screen stocks, get company overviews, run backtests, or macro snapshots.
              Example: &quot;Run DCF on AAPL at 10% WACC&quot;
            </p>
          )}
          {lastAiQuery != null && messages.length === 0 && (
            <div className="space-y-2">
              <div className="flex gap-2">
                <span className="text-primary font-bold">&gt;</span>
                <span className="text-on-surface">{lastAiQuery.q}</span>
              </div>
              <div className="flex gap-2 text-on-surface-variant">
                <span className="text-accent font-bold">~</span>
                <span>
                  {loading ? "Processing..." : lastAiQuery.a || "—"}
                </span>
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className="space-y-2">
              <div className="flex gap-2">
                <span className={m.role === "user" ? "text-primary font-bold" : "text-accent font-bold"}>
                  {m.role === "user" ? ">" : "~"}
                </span>
                <span className={m.role === "user" ? "text-on-surface" : "text-on-surface-variant"}>{m.content}</span>
              </div>
              {m.toolCalls && m.toolCalls.length > 0 && (
                <div className="pl-4 border-l border-outline-variant space-y-1">
                  {m.toolCalls.map((tc, j) => (
                    <div key={j} className="text-[10px] text-tertiary flex items-center gap-2">
                      <span className="material-symbols-outlined text-[12px]">build</span>
                      <span>
                        {tc.tool === "run_dcf" && Boolean(tc.input?.symbol) && `Running DCF on ${String(tc.input.symbol)}…`}
                        {tc.tool === "screen_stocks" && "Running screener…"}
                        {tc.tool === "get_company_overview" && Boolean(tc.input?.symbol) && `Getting overview for ${String(tc.input.symbol)}…`}
                        {tc.tool === "run_backtest" && "Running backtest…"}
                        {tc.tool === "get_macro_snapshot" && "Getting macro snapshot…"}
                        {!["run_dcf", "screen_stocks", "get_company_overview", "run_backtest", "get_macro_snapshot"].includes(tc.tool) && `${tc.tool}`}
                      </span>
                      {tc.status === "complete" && <span className="text-accent-green material-symbols-outlined text-[12px]">check</span>}
                    </div>
                  ))}
                </div>
              )}
              {m.structuredData && Object.keys(m.structuredData).length > 0 && (
                <div className="pl-4 mt-2 space-y-2">
                  {m.structuredData.run_dcf && !("error" in m.structuredData.run_dcf) && (
                    <div className="bg-background border border-outline-variant p-2">
                      <div className="text-accent uppercase text-[10px] mb-1">DCF Result</div>
                      <div>
                        Intrinsic: ${String((m.structuredData.run_dcf as Record<string, unknown>).intrinsic_value_per_share)}
                        {" | "}
                        Current: ${String((m.structuredData.run_dcf as Record<string, unknown>).current_price)}
                        {" | "}
                        Upside: {String((m.structuredData.run_dcf as Record<string, unknown>).upside_downside_pct)}%
                      </div>
                    </div>
                  )}
                  {m.structuredData.screen_stocks && !("error" in m.structuredData.screen_stocks) && (
                    <div className="bg-background border border-outline-variant p-2">
                      <div className="text-accent uppercase text-[10px] mb-1">
                        Screener Results (top {String((m.structuredData.screen_stocks as Record<string, unknown>).count)})
                      </div>
                      <div>
                        {((m.structuredData.screen_stocks as Record<string, unknown>).tickers as Array<Record<string, unknown>>)?.slice(0, 5).map((t, k) => (
                          <div key={k}>
                            {String(t.symbol)}: {String(t.name)} | MCap: {t.market_cap != null ? `$${Number(t.market_cap) / 1e9}B` : "—"} | P/E: {String(t.pe ?? "—")}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {m.structuredData.get_company_overview && !("error" in m.structuredData.get_company_overview) && (
                    <div className="bg-background border border-outline-variant p-2">
                      <div className="text-accent uppercase text-[10px] mb-1">Company Overview</div>
                      <div>
                        {String((m.structuredData.get_company_overview as Record<string, unknown>).name)} | Price: ${String((m.structuredData.get_company_overview as Record<string, unknown>).price)} | Sector: {String((m.structuredData.get_company_overview as Record<string, unknown>).sector)}
                      </div>
                    </div>
                  )}
                  {m.structuredData.run_backtest && !("error" in m.structuredData.run_backtest) && (
                    <div className="bg-background border border-outline-variant p-2">
                      <div className="text-accent uppercase text-[10px] mb-1">Backtest Result</div>
                      <div>
                        Sharpe: {String((m.structuredData.run_backtest as Record<string, unknown>).sharpe_ratio)} | CAGR: {String((m.structuredData.run_backtest as Record<string, unknown>).cagr_pct)}% | Max DD: {String((m.structuredData.run_backtest as Record<string, unknown>).max_drawdown_pct)}%
                      </div>
                    </div>
                  )}
                  {m.structuredData.get_macro_snapshot && !("error" in m.structuredData.get_macro_snapshot) && (
                    <div className="bg-background border border-outline-variant p-2">
                      <div className="text-accent uppercase text-[10px] mb-1">Macro Snapshot</div>
                      <div>
                        {JSON.stringify(m.structuredData.get_macro_snapshot)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {loading && toolCallsActive.length === 0 && messages.length > 0 && (
            <div className="text-on-surface-variant animate-pulse">Processing...</div>
          )}
        </div>
        <div className="flex-shrink-0">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendMessage(input, false);
            }}
            className="flex gap-2 w-full"
          >
            <input
              className="flex-1 bg-background border border-outline-variant text-on-surface px-3 py-2 font-data-mono text-[12px] uppercase focus:outline-none focus:border-on-tertiary-container transition-all"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about stocks, DCF, screener…"
              disabled={loading}
            />
            <button type="submit" className="text-label-xs font-label-xs uppercase tracking-widest px-4 py-2 border border-outline-variant hover:bg-background transition-colors text-on-surface disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading}>
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
