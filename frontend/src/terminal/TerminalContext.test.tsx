import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import React from "react";
import { TerminalContext, useTerminal, MODULES_ORDER } from "./TerminalContext";
import type { ActiveModule } from "./TerminalContext";

function createWrapper(initialSymbol = "AAPL", initialModule: ActiveModule = "primary") {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    const [primarySymbol, setPrimarySymbol] = React.useState(initialSymbol);
    const [activeModule, setActiveModule] = React.useState(initialModule);
    const [lastAiQuery, setLastAiQuery] = React.useState<{ q: string; a: string } | null>(null);
    const [wsConnected, setWsConnected] = React.useState(false);
    const [lastBacktestSymbol, setLastBacktestSymbol] = React.useState<string | null>(null);
    const value = {
      primarySymbol,
      setPrimarySymbol,
      activeModule,
      setActiveModule,
      lastAiQuery,
      setLastAiQuery: setLastAiQuery as (v: { q: string; a: string } | null) => void,
      setAiResponse: () => {},
      wsConnected,
      setWsConnected,
      onRunBacktest: (sym: string) => setLastBacktestSymbol(sym),
      lastBacktestSymbol,
      onSwitchWorkspace: () => {},
      nextModule: () => {
        const i = MODULES_ORDER.indexOf(activeModule);
        setActiveModule(MODULES_ORDER[(i + 1) % MODULES_ORDER.length]!);
      },
      prevModule: () => {
        const i = MODULES_ORDER.indexOf(activeModule);
        setActiveModule(MODULES_ORDER[(i - 1 + MODULES_ORDER.length) % MODULES_ORDER.length]!);
      },
    };
    return (
      <TerminalContext.Provider value={value}>
        {children}
      </TerminalContext.Provider>
    );
  };
}

describe("useTerminal", () => {
  it("throws outside provider", () => {
    expect(() => renderHook(() => useTerminal())).toThrow("useTerminal must be used within");
  });

  it("returns initial primarySymbol and activeModule", () => {
    const wrapper = createWrapper("MSFT", "fundamental");
    const { result } = renderHook(() => useTerminal(), { wrapper });
    expect(result.current.primarySymbol).toBe("MSFT");
    expect(result.current.activeModule).toBe("fundamental");
  });

  it("setPrimarySymbol and setActiveModule update state", () => {
    const wrapper = createWrapper("AAPL", "primary");
    const { result } = renderHook(() => useTerminal(), { wrapper });
    act(() => {
      result.current.setPrimarySymbol("GOOGL");
    });
    expect(result.current.primarySymbol).toBe("GOOGL");
    act(() => {
      result.current.setActiveModule("quant");
    });
    expect(result.current.activeModule).toBe("quant");
  });

  it("nextModule and prevModule cycle modules", () => {
    const wrapper = createWrapper("AAPL", "primary");
    const { result } = renderHook(() => useTerminal(), { wrapper });
    act(() => {
      result.current.nextModule();
    });
    expect(result.current.activeModule).toBe("fundamental");
    act(() => {
      result.current.prevModule();
    });
    expect(result.current.activeModule).toBe("primary");
  });
});
