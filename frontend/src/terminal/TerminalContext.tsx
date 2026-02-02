import React from "react";

export type ActiveModule =
  | "primary"
  | "fundamental"
  | "technical"
  | "quant"
  | "economic"
  | "news"
  | "portfolio"
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
  { code: "SCREEN", desc: "Screening & discovery" },
  { code: "AI [query]", desc: "AI assistant (or type freely)" },
  { code: "BACKTEST [ticker]", desc: "Run strategy backtest" },
  { code: "WORKSPACE [name]", desc: "Switch workspace" },
  { code: "? or HELP", desc: "Show this help" },
];

export const MODULES_ORDER: ActiveModule[] = [
  "primary",
  "fundamental",
  "technical",
  "quant",
  "economic",
  "news",
  "portfolio",
  "screening",
  "ai",
];

export const WORKSPACE_STORAGE_KEY = "bloomberg-workspaces";
export const ACTIVE_WORKSPACE_KEY = "bloomberg-active-workspace";

export interface WorkspaceState {
  name: string;
  layout: [number, number, number];
  activeModule: ActiveModule;
  primarySymbol?: string;
}
