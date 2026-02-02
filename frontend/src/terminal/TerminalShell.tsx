import React, { useState, useCallback } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { MarketOverview } from "./panels/MarketOverview";
import { PrimaryInstrument } from "./panels/PrimaryInstrument";
import { PortfolioPanel } from "./panels/PortfolioPanel";
import { AiAssistantPanel } from "./panels/AiAssistantPanel";
import { FundamentalPanel } from "./panels/FundamentalPanel";
import { TechnicalPanel } from "./panels/TechnicalPanel";
import { QuantPanel } from "./panels/QuantPanel";
import { EconomicPanel } from "./panels/EconomicPanel";
import { NewsPanel } from "./panels/NewsPanel";
import { ScreeningPanel } from "./panels/ScreeningPanel";
import { PaperTradingPanel } from "./panels/PaperTradingPanel";
import { AutomationPanel } from "./panels/AutomationPanel";
import { CommandBar } from "./CommandBar";
import { TickerStrip } from "./TickerStrip";
import {
  TerminalContext,
  type ActiveModule,
  MODULES_ORDER,
  WORKSPACE_STORAGE_KEY,
  ACTIVE_WORKSPACE_KEY,
  WATCHLIST_STORAGE_KEY,
  loadWatchlist,
  type WorkspaceState,
} from "./TerminalContext";

const MODULE_LABELS: Record<ActiveModule, string> = {
  primary: "Primary",
  fundamental: "Fundamental",
  technical: "Technical",
  quant: "Quant",
  economic: "Economic",
  news: "News",
  portfolio: "Portfolio",
  paper: "Paper",
  automation: "Automation",
  screening: "Screening",
  ai: "AI",
};

const LAYOUT_STORAGE_KEY = "bloomberg-terminal-layout";
const DEFAULT_LAYOUT: [number, number, number] = [20, 50, 30];

function loadWorkspaces(): Record<string, WorkspaceState> {
  try {
    const raw = localStorage.getItem(WORKSPACE_STORAGE_KEY);
    if (!raw) return {};
    const obj = JSON.parse(raw) as unknown;
    return typeof obj === "object" && obj !== null ? (obj as Record<string, WorkspaceState>) : {};
  } catch {
    return {};
  }
}

function saveWorkspaces(workspaces: Record<string, WorkspaceState>) {
  try {
    localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(workspaces));
  } catch {
    // ignore
  }
}

function loadLayout(): [number, number, number] | undefined {
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) return undefined;
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr) || arr.length !== 3) return undefined;
    const [a, b, c] = arr.map(Number);
    if (a + b + c !== 100 || a < 10 || b < 20 || c < 10) return undefined;
    return [a, b, c];
  } catch {
    return undefined;
  }
}

function saveLayout(layout: { [id: string]: number }) {
  try {
    const left = layout.left ?? 20;
    const main = layout.main ?? 50;
    const right = layout.right ?? 30;
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify([left, main, right]));
  } catch {
    // ignore
  }
}

function MainContent({ activeModule }: { activeModule: ActiveModule }) {
  switch (activeModule) {
    case "primary":
      return (
        <>
          <PrimaryInstrument />
          <PortfolioPanel />
        </>
      );
    case "fundamental":
      return <FundamentalPanel />;
    case "technical":
      return <TechnicalPanel />;
    case "quant":
      return <QuantPanel />;
    case "economic":
      return <EconomicPanel />;
    case "news":
      return <NewsPanel />;
    case "portfolio":
      return <PortfolioPanel />;
    case "paper":
      return <PaperTradingPanel />;
    case "automation":
      return <AutomationPanel />;
    case "screening":
      return <ScreeningPanel />;
    case "ai":
      return (
        <>
          <PrimaryInstrument />
          <PortfolioPanel />
        </>
      );
    default:
      return (
        <>
          <PrimaryInstrument />
          <PortfolioPanel />
        </>
      );
  }
}

export const TerminalShell: React.FC = () => {
  const [currentWorkspaceName, setCurrentWorkspaceName] = useState(() => {
    try {
      return localStorage.getItem(ACTIVE_WORKSPACE_KEY) || "Default";
    } catch {
      return "Default";
    }
  });
  const savedLayout = loadLayout();
  const workspacesInitial = loadWorkspaces();
  const currentWs = workspacesInitial[currentWorkspaceName];
  const [primarySymbol, setPrimarySymbol] = useState(currentWs?.primarySymbol ?? "AAPL");
  const [activeModule, setActiveModule] = useState<ActiveModule>(currentWs?.activeModule ?? "primary");
  const [lastAiQuery, setLastAiQuery] = useState<{ q: string; a: string } | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastBacktestSymbol, setLastBacktestSymbol] = useState<string | null>(null);
  const [watchlist, setWatchlistState] = useState<string[]>(loadWatchlist);
  const setWatchlist = useCallback((symbols: string[]) => {
    setWatchlistState(symbols);
    try {
      localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(symbols));
    } catch {
      // ignore
    }
  }, []);
  const defaultLayout = currentWs?.layout
    ? { left: currentWs.layout[0], main: currentWs.layout[1], right: currentWs.layout[2] }
    : savedLayout
      ? { left: savedLayout[0], main: savedLayout[1], right: savedLayout[2] }
      : { left: DEFAULT_LAYOUT[0], main: DEFAULT_LAYOUT[1], right: DEFAULT_LAYOUT[2] };

  const handleCommand = useCallback((input: string) => {
    setLastAiQuery({ q: input.trim(), a: "" });
  }, []);

  const setAiResponse = useCallback((a: string) => {
    setLastAiQuery((prev) => (prev ? { ...prev, a } : null));
  }, []);

  const handleRunBacktest = useCallback((symbol: string) => {
    setLastBacktestSymbol(symbol);
    setActiveModule("quant");
  }, []);

  const handleLayoutChanged = useCallback((layout: { [id: string]: number }) => {
    const left = layout.left ?? 20;
    const main = layout.main ?? 50;
    const right = layout.right ?? 30;
    saveLayout(layout);
    const workspaces = loadWorkspaces();
    workspaces[currentWorkspaceName] = {
      name: currentWorkspaceName,
      layout: [left, main, right],
      activeModule,
      primarySymbol,
    };
    saveWorkspaces(workspaces);
  }, [currentWorkspaceName, activeModule, primarySymbol]);

  const handleSwitchWorkspace = useCallback((name: string) => {
    const workspaces = loadWorkspaces();
    const currentLayout = workspaces[currentWorkspaceName]?.layout ?? [defaultLayout.left, defaultLayout.main, defaultLayout.right];
    const currentState: WorkspaceState = {
      name: currentWorkspaceName,
      layout: currentLayout,
      activeModule,
      primarySymbol,
    };
    workspaces[currentWorkspaceName] = currentState;
    saveWorkspaces(workspaces);
    try {
      localStorage.setItem(ACTIVE_WORKSPACE_KEY, name);
    } catch {
      // ignore
    }
    setCurrentWorkspaceName(name);
    const next = workspaces[name] ?? {
      name,
      layout: DEFAULT_LAYOUT,
      activeModule: "primary" as ActiveModule,
      primarySymbol: "AAPL",
    };
    setActiveModule(next.activeModule);
    setPrimarySymbol(next.primarySymbol ?? "AAPL");
  }, [currentWorkspaceName, activeModule, primarySymbol, defaultLayout.left, defaultLayout.main, defaultLayout.right]);

  const nextModule = useCallback(() => {
    const idx = MODULES_ORDER.indexOf(activeModule);
    const next = MODULES_ORDER[(idx + 1) % MODULES_ORDER.length];
    setActiveModule(next ?? "primary");
  }, [activeModule]);

  const prevModule = useCallback(() => {
    const idx = MODULES_ORDER.indexOf(activeModule);
    const next = MODULES_ORDER[idx <= 0 ? MODULES_ORDER.length - 1 : idx - 1];
    setActiveModule(next ?? "primary");
  }, [activeModule]);

  return (
    <TerminalContext.Provider
      value={{
        primarySymbol,
        setPrimarySymbol,
        activeModule,
        setActiveModule,
        lastAiQuery,
        setLastAiQuery,
        setAiResponse,
        wsConnected,
        setWsConnected,
        onRunBacktest: handleRunBacktest,
        lastBacktestSymbol,
        onSwitchWorkspace: handleSwitchWorkspace,
        nextModule,
        prevModule,
        watchlist,
        setWatchlist,
      }}
    >
      <div className="terminal-root">
        <header className="terminal-header">
          <div className="terminal-title">BLOOMBERG</div>
          <div className="terminal-status">
            <span className={`terminal-status-dot ${wsConnected ? "live" : ""}`} aria-hidden />
            {wsConnected ? "Live" : "Connecting…"} • /api
          </div>
        </header>

        <div className="terminal-command-row" style={{ position: "relative" }}>
          <CommandBar onSubmit={handleCommand} />
        </div>

        <nav
          className="terminal-module-tabs"
          aria-label="Switch module"
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 4,
            padding: "6px 8px",
            borderBottom: "1px solid var(--border)",
            background: "var(--bg-panel)",
            minHeight: 36,
            alignItems: "center",
          }}
        >
          {MODULES_ORDER.map((mod) => (
            <button
              key={mod}
              type="button"
              onClick={() => setActiveModule(mod)}
              className={activeModule === mod ? "terminal-tab-active" : "terminal-tab"}
              style={{
                padding: "4px 10px",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
                border: `1px solid ${activeModule === mod ? "var(--accent)" : "var(--border)"}`,
                borderRadius: 4,
                background: activeModule === mod ? "var(--accent)" : "transparent",
                color: activeModule === mod ? "#0a0a0a" : "var(--text)",
                cursor: "pointer",
              }}
            >
              {MODULE_LABELS[mod]}
            </button>
          ))}
        </nav>

        <TickerStrip primarySymbol={primarySymbol} onWsStatus={setWsConnected} />

        <Group
          key={currentWorkspaceName}
          orientation="horizontal"
          className="terminal-panels"
          defaultLayout={defaultLayout}
          onLayoutChanged={handleLayoutChanged}
        >
          <Panel id="left" defaultSize={defaultLayout.left} minSize={15} maxSize={35} style={{ minWidth: 0, overflow: "auto" }}>
            <div style={{ padding: 8, height: "100%" }}>
              <MarketOverview />
            </div>
          </Panel>
          <Separator id="left-sep" style={{ width: 4, background: "var(--border)", cursor: "col-resize" }} />
          <Panel id="main" defaultSize={defaultLayout.main} minSize={30} maxSize={70} style={{ minWidth: 0, overflow: "auto" }}>
            <div style={{ padding: 8, height: "100%", display: "flex", flexDirection: "column", gap: 8 }}>
              <MainContent activeModule={activeModule} />
            </div>
          </Panel>
          <Separator id="right-sep" style={{ width: 4, background: "var(--border)", cursor: "col-resize" }} />
          <Panel id="right" defaultSize={defaultLayout.right} minSize={20} maxSize={45} style={{ minWidth: 0, overflow: "auto" }}>
            <div style={{ padding: 8, height: "100%" }}>
              <AiAssistantPanel />
            </div>
          </Panel>
        </Group>
      </div>
    </TerminalContext.Provider>
  );
};

