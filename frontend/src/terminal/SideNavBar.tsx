import React from "react";
import { useTerminal } from "./TerminalContext";

export const SideNavBar: React.FC = () => {
  const { activeModule, setActiveModule } = useTerminal();

  const navItems = [
    { id: "primary", icon: "trending_up", label: "EQUITIES" },
    { id: "technical", icon: "insights", label: "TECHNICAL" },
    { id: "quant", icon: "functions", label: "QUANT" },
    { id: "fundamental", icon: "description", label: "RESEARCH" },
  ] as const;

  return (
    <aside className="fixed left-0 top-20 h-[calc(100vh-80px)] w-20 flex flex-col items-center py-6 z-40 bg-background dark:bg-background border-r border-outline-variant">
      <div className="flex flex-col items-center gap-8 flex-grow w-full">
        <div className="w-10 h-10 bg-surface-container-high flex items-center justify-center mb-4 hairline-border">
          <span className="material-symbols-outlined text-on-surface">grid_view</span>
        </div>
        
        {navItems.map((item) => {
          const isActive = activeModule === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveModule(item.id)}
              className={`w-full py-4 flex flex-col items-center gap-1 transition-none ${
                isActive 
                  ? "text-on-surface dark:text-on-surface border-r-2 border-primary bg-surface-container-highest"
                  : "text-outline dark:text-outline hover:text-on-surface hover:bg-surface-container-low"
              }`}
            >
              <span className="material-symbols-outlined">{item.icon}</span>
              <span className="font-label-xs text-[8px] uppercase tracking-tighter">{item.label}</span>
            </button>
          );
        })}
      </div>
      <div className="flex flex-col items-center gap-4 mt-auto">
        <button className="material-symbols-outlined text-outline hover:text-on-surface transition-none">help</button>
        <button className="material-symbols-outlined text-outline hover:text-on-surface transition-none">logout</button>
      </div>
    </aside>
  );
};
