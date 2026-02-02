import React, { useState, useRef, useEffect, useCallback } from "react";
import { useTerminal, COMMAND_HELP } from "./TerminalContext";
import { parseCommand } from "./parseCommand";

const HISTORY_KEY = "bloomberg-command-history";
const MAX_HISTORY = 10;

function loadHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as unknown;
    return Array.isArray(arr) ? arr.slice(0, MAX_HISTORY) : [];
  } catch {
    return [];
  }
}

function saveHistory(history: string[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
  } catch {
    // ignore
  }
}

interface CommandBarProps {
  onSubmit: (input: string) => void;
}

export const CommandBar: React.FC<CommandBarProps> = ({ onSubmit }) => {
  const {
    primarySymbol,
    setPrimarySymbol,
    setActiveModule,
    onRunBacktest,
    onSwitchWorkspace,
    nextModule,
    prevModule,
  } = useTerminal();
  const [value, setValue] = useState("");
  const [history, setHistory] = useState<string[]>(loadHistory);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showHelp, setShowHelp] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pushHistory = useCallback((entry: string) => {
    const trimmed = entry.trim();
    if (!trimmed) return;
    setHistory((prev) => {
      const next = [trimmed, ...prev.filter((x) => x !== trimmed)].slice(0, MAX_HISTORY);
      saveHistory(next);
      return next;
    });
    setHistoryIndex(-1);
  }, []);

  const handleParsedCommand = useCallback(
    (raw: string) => {
      const parsed = parseCommand(raw);
      if (!parsed) return;
      pushHistory(raw);
      setValue("");

      if (parsed.type === "help") {
        setShowHelp(true);
        return;
      }
      if (parsed.module) setActiveModule(parsed.module);
      if (parsed.symbol) setPrimarySymbol(parsed.symbol);
      if (parsed.type === "ai" && parsed.query != null) onSubmit(parsed.query);
      if (parsed.type === "backtest") onRunBacktest(parsed.symbol || primarySymbol);
      if (parsed.type === "workspace" && parsed.symbol) onSwitchWorkspace(parsed.symbol);
      if (parsed.type === "ai" && parsed.query != null) onSubmit(parsed.query);
    },
    [
      setPrimarySymbol,
      setActiveModule,
      onRunBacktest,
      onSwitchWorkspace,
      primarySymbol,
      pushHistory,
      onSubmit,
    ]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const v = value.trim();
      if (!v) return;
      handleParsedCommand(v);
    },
    [value, handleParsedCommand]
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const isInput =
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA";
      if (
        (e.key === "/" || (e.key === "k" && (e.metaKey || e.ctrlKey))) &&
        !isInput
      ) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") {
        setShowHelp(false);
        setValue("");
        setHistoryIndex(-1);
        inputRef.current?.blur();
      }
      if (!isInput && e.altKey) {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          nextModule();
        }
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          prevModule();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [nextModule, prevModule]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSubmit(e as unknown as React.FormEvent);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (history.length === 0) return;
      const next =
        historyIndex < history.length - 1 ? historyIndex + 1 : history.length - 1;
      setHistoryIndex(next);
      setValue(history[next] ?? "");
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (historyIndex <= 0) {
        setHistoryIndex(-1);
        setValue("");
        return;
      }
      const next = historyIndex - 1;
      setHistoryIndex(next);
      setValue(history[next] ?? "");
    }
  };

  return (
    <>
      <form onSubmit={handleSubmit} style={{ flex: 1, display: "flex", minWidth: 0 }}>
        <input
          ref={inputRef}
          type="text"
          className="terminal-command-input"
          placeholder="Ticker or command (e.g. AAPL, FA AAPL, ECO, ? for help)"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Command or ticker"
        />
      </form>
      {showHelp && (
        <div
          className="command-help-overlay"
          role="dialog"
          aria-label="Command help"
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            marginTop: 4,
            background: "var(--bg-panel)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            padding: "12px 16px",
            zIndex: 1000,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            maxHeight: 320,
            overflowY: "auto",
          }}
        >
          <div style={{ color: "var(--accent)", marginBottom: 8, fontWeight: 600 }}>
            Commands
          </div>
          {COMMAND_HELP.map(({ code, desc }) => (
            <div
              key={code}
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 16,
                padding: "4px 0",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <span style={{ color: "var(--text)" }}>{code}</span>
              <span style={{ color: "var(--text-soft)" }}>{desc}</span>
            </div>
          ))}
          <div style={{ color: "var(--text-soft)", marginTop: 12, marginBottom: 4, fontSize: 11 }}>
            Shortcuts: / or Cmd+K focus bar · Alt+Left/Right switch module · Esc clear
          </div>
          <button
            type="button"
            className="ai-button"
            style={{ marginTop: 12 }}
            onClick={() => setShowHelp(false)}
          >
            Close
          </button>
        </div>
      )}
    </>
  );
};
