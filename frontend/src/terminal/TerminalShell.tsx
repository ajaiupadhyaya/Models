import React from "react";
import { MarketOverview } from "./panels/MarketOverview";
import { PrimaryInstrument } from "./panels/PrimaryInstrument";
import { PortfolioPanel } from "./panels/PortfolioPanel";
import { AiAssistantPanel } from "./panels/AiAssistantPanel";

export const TerminalShell: React.FC = () => {
  return (
    <div className="terminal-root">
      <header className="terminal-header">
        <div className="terminal-title">Local Bloomberg Terminal</div>
        <div className="terminal-status">Backend: /api • FastAPI</div>
      </header>

      <div className="terminal-layout">
        <aside className="terminal-sidebar">
          <MarketOverview />
        </aside>

        <main className="terminal-main">
          <PrimaryInstrument />
          <PortfolioPanel />
        </main>

        <aside className="terminal-right">
          <AiAssistantPanel />
        </aside>
      </div>
    </div>
  );
};

