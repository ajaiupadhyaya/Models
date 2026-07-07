import React from "react";
import { useTerminal } from "./TerminalContext";

export const TopNavBar: React.FC = () => {
  const { activeModule, setActiveModule, primarySymbol, setPrimarySymbol } = useTerminal();

  return (
    <header className="fixed top-0 left-0 w-full z-50 h-20 bg-background dark:bg-background border-b border-outline-variant flex justify-between items-center px-margin-page">
      <div className="flex items-center gap-12">
        <div className="font-headline-lg text-headline-lg font-black tracking-tighter text-on-background dark:text-on-background">
          FIN-TERMINAL
        </div>
        <nav className="hidden md:flex gap-8">
          {(["primary", "portfolio", "news"] as const).map((mod) => (
            <button
              key={mod}
              onClick={() => setActiveModule(mod)}
              className={`font-label-xs text-label-xs uppercase tracking-widest px-1 transition-colors duration-100 ${
                activeModule === mod
                  ? "text-primary dark:text-primary font-bold border-b-2 border-primary"
                  : "text-on-surface-variant dark:text-on-surface-variant hover:bg-on-surface hover:text-background"
              }`}
            >
              {mod}
            </button>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-6">
        <div className="relative group">
          <input
            className="bg-surface-container-low border border-outline-variant px-4 py-2 font-data-mono text-data-mono w-64 focus:outline-none focus:border-on-tertiary-container transition-all text-on-surface"
            placeholder="SEARCH TICKER..."
            type="text"
            value={primarySymbol}
            onChange={(e) => setPrimarySymbol(e.target.value.toUpperCase())}
          />
          <span className="absolute right-3 top-2.5 material-symbols-outlined text-outline text-sm">search</span>
        </div>
        <div className="flex gap-4">
          <button className="material-symbols-outlined text-on-surface hover:text-primary transition-colors">settings</button>
          <button className="material-symbols-outlined text-on-surface hover:text-primary transition-colors">account_circle</button>
        </div>
      </div>
    </header>
  );
};
