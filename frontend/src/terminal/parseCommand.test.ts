import { describe, it, expect } from "vitest";
import { parseCommand } from "./parseCommand";

describe("parseCommand", () => {
  it("parses HELP and ? as help", () => {
    expect(parseCommand("?")).toEqual({ type: "help" });
    expect(parseCommand("HELP")).toEqual({ type: "help" });
  });

  it("parses single ticker as primary module", () => {
    expect(parseCommand("AAPL")).toEqual({ type: "module", module: "primary", symbol: "AAPL" });
    expect(parseCommand("MSFT")).toEqual({ type: "module", module: "primary", symbol: "MSFT" });
  });

  it("parses GP as primary with optional ticker", () => {
    expect(parseCommand("GP AAPL")).toEqual({ type: "module", module: "primary", symbol: "AAPL" });
    expect(parseCommand("gp")).toEqual({ type: "module", module: "primary" });
  });

  it("parses FA as fundamental with optional ticker", () => {
    expect(parseCommand("FA MSFT")).toEqual({ type: "module", module: "fundamental", symbol: "MSFT" });
  });

  it("parses FLDS/FLD as technical", () => {
    expect(parseCommand("FLDS AAPL")).toEqual({ type: "module", module: "technical", symbol: "AAPL" });
    expect(parseCommand("FLD")).toEqual({ type: "module", module: "technical" });
  });

  it("parses ECO as economic", () => {
    expect(parseCommand("ECO")).toEqual({ type: "module", module: "economic" });
  });

  it("parses N as news with optional ticker", () => {
    expect(parseCommand("N AAPL")).toEqual({ type: "module", module: "news", symbol: "AAPL" });
  });

  it("parses PORT as portfolio", () => {
    expect(parseCommand("PORT")).toEqual({ type: "module", module: "portfolio" });
  });

  it("parses SCREEN as screening", () => {
    expect(parseCommand("SCREEN")).toEqual({ type: "module", module: "screening" });
  });

  it("parses BACKTEST as backtest with quant module", () => {
    expect(parseCommand("BACKTEST AAPL")).toEqual({ type: "backtest", module: "quant", symbol: "AAPL" });
  });

  it("parses WORKSPACE with optional name", () => {
    expect(parseCommand("WORKSPACE myws")).toEqual({ type: "workspace", symbol: "MYWS" });
  });

  it("parses AI with query", () => {
    expect(parseCommand("AI what is AAPL")).toEqual({ type: "ai", module: "ai", query: "what is AAPL" });
  });

  it("parses unknown command as ai with full query", () => {
    const r = parseCommand("random query");
    expect(r?.type).toBe("ai");
    expect(r?.module).toBe("ai");
    expect(r?.query).toBe("random query");
  });
});
