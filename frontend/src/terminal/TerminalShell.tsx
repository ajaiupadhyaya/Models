import React, { useState, useCallback, useEffect } from "react";
import { getApiBase, resolveApiUrl } from "../apiBase";
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
import { AiInsightsPanel } from "./panels/AiInsightsPanel";
import { DataStatusPanel } from "./panels/DataStatusPanel";
import { BacktestPanel } from "./panels/BacktestPanel";
import { OptimizerPanel } from "./panels/OptimizerPanel";
import { StressTestPanel } from "./panels/StressTestPanel";
import { NewsSentimentPanel } from "./panels/NewsSentimentPanel";
import { CommandBar } from "./CommandBar";
import { TickerSearchBar } from "./TickerSearchBar";
import { TickerStrip } from "./TickerStrip";
import { TopNavBar } from "./TopNavBar";
import { SideNavBar } from "./SideNavBar";
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
  dataStatus: "Data Status",
  primary: "Primary",
  fundamental: "Fundamental",
  technical: "Technical",
  quant: "Quant",
  economic: "Economic",
  news: "News",
  newsSentiment: "News & Sentiment",
  portfolio: "Portfolio",
  backtest: "Backtest",
  optimizer: "Optimizer",
  stressTest: "Stress Test",
  paper: "Paper",
  automation: "Automation",
  screening: "Screening",
  ai: "AI",
};

const LAYOUT_STORAGE_KEY = "bloomberg-terminal-layout";
const DEFAULT_LAYOUT: [number, number, number] = [20, 50, 30];

function getApiLabel(): string {
  const base = getApiBase();
  if (!base) return "/api";
  try {
    return new URL(base).host;
  } catch {
    return base;
  }
}

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
    case "dataStatus":
      return <DataStatusPanel />;
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
    case "newsSentiment":
      return <NewsSentimentPanel />;
    case "portfolio":
      return <PortfolioPanel />;
    case "backtest":
      return <BacktestPanel />;
    case "optimizer":
      return <OptimizerPanel />;
    case "stressTest":
      return <StressTestPanel />;
    case "paper":
      return <PaperTradingPanel />;
    case "automation":
      return <AutomationPanel />;
    case "screening":
      return <ScreeningPanel />;
    case "ai":
      return (
        <>
          <AiInsightsPanel />
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
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);
  const apiLabel = getApiLabel();

  const checkApiHealth = useCallback(() => {
    fetch(resolveApiUrl("/health"))
      .then((r) => setApiHealthy(r.ok))
      .catch(() => setApiHealthy(false));
  }, []);
  useEffect(() => {
    checkApiHealth();
  }, [checkApiHealth]);

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
    setPrimarySymbol(symbol);
    setActiveModule("backtest");
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

  if (apiHealthy === null) {
    return (
      <div className="terminal-state-screen">
        <p className="login-subtitle">Checking API…</p>
      </div>
    );
  }

  if (apiHealthy === false) {
    return (
      <div className="terminal-state-screen terminal-state-screen-gap">
        <h2 className="login-title terminal-state-title">API unreachable</h2>
        <p className="login-subtitle terminal-state-message">
          The backend is not responding. Start the API locally or verify your deployment. Check that{" "}
          <code>/health</code> returns 200.
        </p>
        <button type="button" className="ai-button" onClick={checkApiHealth}>
          Retry
        </button>
      </div>
    );
  }

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
      <div className="min-h-screen bg-background text-on-surface flex flex-col">
        <TopNavBar />
        <SideNavBar />
        
        <main className="ml-20 mt-16 p-margin-page flex-1 flex flex-col bg-outline-variant gap-gutter relative z-0 h-[calc(100vh-64px)] overflow-hidden">
          <div className="bg-background hairline-b shrink-0 flex flex-col">
            <TickerStrip primarySymbol={primarySymbol} onWsStatus={setWsConnected} />
            <div className="px-4 py-2 border-b border-outline-variant">
              <CommandBar onSubmit={handleCommand} />
            </div>
          </div>
          
          <Group
            key={currentWorkspaceName}
            orientation="horizontal"
            className="flex-1 min-h-0 bg-background"
            defaultLayout={defaultLayout}
            onLayoutChanged={handleLayoutChanged}
          >
            <Panel id="left" defaultSize={defaultLayout.left} minSize={15} maxSize={35} className="min-w-0 overflow-auto bg-surface-container-low hairline-r">
              <div className="h-full p-4">
                <MarketOverview />
              </div>
            </Panel>
            <Separator id="left-sep" className="w-[1px] bg-outline-variant cursor-col-resize hover:bg-primary transition-colors" />
            <Panel id="main" defaultSize={defaultLayout.main} minSize={30} maxSize={70} className="min-w-0 overflow-y-auto bg-background">
              <div className="min-h-full min-w-0 flex flex-col">
                <MainContent activeModule={activeModule} />
              </div>
            </Panel>
            <Separator id="right-sep" className="w-[1px] bg-outline-variant cursor-col-resize hover:bg-primary transition-colors" />
            <Panel id="right" defaultSize={defaultLayout.right} minSize={20} maxSize={45} className="min-w-0 overflow-auto bg-surface-container-low hairline-l">
              <div className="h-full p-4">
                <AiAssistantPanel />
              </div>
            </Panel>
          </Group>
        </main>
      </div>
    </TerminalContext.Provider>
  );
};
