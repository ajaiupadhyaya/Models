import React from "react";

export type ActiveModule =
  | "primary"
  | "fundamental"
  | "technical"
  | "quant"
  | "economic"
  | "news"
  | "portfolio"
  | "paper"
  | "automation"
  | "screening"
  | "ai";

export interface TerminalContextValue {
  primarySymbol: string;
  setPrimarySymbol: (s: string) => void;
  activeModule: ActiveModule;
  setActiveModule: (m: ActiveModule) => void;
  lastAiQuery: { q: string; a: string } | null;
  setLastAiQuery: (v: { q: string; a: string } | null) => void;
  setAiResponse: (a: string) => void;
  wsConnected: boolean;
  setWsConnected: (v: boolean) => void;
  onRunBacktest: (symbol: string) => void;
  lastBacktestSymbol: string | null;
  onSwitchWorkspace: (name: string) => void;
  nextModule: () => void;
  prevModule: () => void;
  watchlist: string[];
  setWatchlist: (symbols: string[]) => void;
}

export const TerminalContext = React.createContext<TerminalContextValue | null>(null);

export function useTerminal(): TerminalContextValue {
  const ctx = React.useContext(TerminalContext);
  if (!ctx) throw new Error("useTerminal must be used within TerminalContext.Provider");
  return ctx;
}

export const COMMAND_HELP: { code: string; desc: string }[] = [
  { code: "GP [ticker]", desc: "Graph / Primary instrument" },
  { code: "FA [ticker]", desc: "Fundamental analysis" },
  { code: "FLDS [ticker]", desc: "Technical / Fields" },
  { code: "ECO", desc: "Economic indicators" },
  { code: "N [ticker]", desc: "News" },
  { code: "PORT", desc: "Portfolio & strategies" },
  { code: "PAPER", desc: "Paper trading" },
  { code: "AUTO / ORCH", desc: "Automation / Orchestrator" },
  { code: "SCREEN", desc: "Screening & discovery" },
  { code: "AI [query]", desc: "AI assistant (or type freely)" },
  { code: "BACKTEST [ticker]", desc: "Run strategy backtest" },
  { code: "TRAIN [ticker]", desc: "Quant / Train model" },
  { code: "WORKSPACE [name]", desc: "Switch workspace" },
  { code: "? or HELP", desc: "Show this help" },
  { code: "/docs", desc: "API docs (Swagger) at /docs when API running" },
];

export const MODULES_ORDER: ActiveModule[] = [
  "primary",
  "fundamental",
  "technical",
  "quant",
  "economic",
  "news",
  "portfolio",
  "paper",
  "automation",
  "screening",
  "ai",
];

export const WORKSPACE_STORAGE_KEY = "bloomberg-workspaces";
export const ACTIVE_WORKSPACE_KEY = "bloomberg-active-workspace";
export const WATCHLIST_STORAGE_KEY = "terminal_watchlist";
const DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"];

export function loadWatchlist(): string[] {
  try {
    const raw = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    if (!raw) return [...DEFAULT_WATCHLIST];
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr)) return [...DEFAULT_WATCHLIST];
    const symbols = arr.filter((x): x is string => typeof x === "string" && x.length > 0);
    return symbols.length > 0 ? symbols : [...DEFAULT_WATCHLIST];
  } catch {
    return [...DEFAULT_WATCHLIST];
  }
}

export interface WorkspaceState {
  name: string;
  layout: [number, number, number];
  activeModule: ActiveModule;
  primarySymbol?: string;
}
