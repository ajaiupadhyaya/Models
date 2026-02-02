/**
 * Parse command bar input into module and symbol.
 * Used by CommandBar and testable in isolation.
 */

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

export interface ParsedCommand {
  type: "help" | "module" | "ai" | "backtest" | "workspace";
  module?: ActiveModule;
  symbol?: string;
  query?: string;
}

export function parseCommand(raw: string): ParsedCommand | null {
  const upper = raw.toUpperCase().trim();
  const parts = raw.trim().split(/\s+/);

  if (upper === "?" || upper === "HELP") {
    return { type: "help" };
  }

  const code = parts[0]?.toUpperCase() ?? "";
  const ticker = parts[1]?.toUpperCase() ?? "";

  // Single-word: could be a command (ECO, PORT, FLD) or a ticker (AAPL)
  const singleWordNoArgCodes = ["ECO", "PORT", "FLD", "FLDS"];
  if (parts.length === 1) {
    if (singleWordNoArgCodes.includes(code)) {
      if (code === "ECO") return { type: "module", module: "economic" };
      if (code === "PORT") return { type: "module", module: "portfolio" };
      if (code === "FLD" || code === "FLDS") return { type: "module", module: "technical" };
    }
    if (/^[A-Z]{1,5}$/.test(parts[0]!)) {
      return { type: "module", module: "primary", symbol: parts[0]! };
    }
  }

  switch (code) {
    case "GP":
      return { type: "module", module: "primary", symbol: ticker || undefined };
    case "FA":
      return { type: "module", module: "fundamental", symbol: ticker || undefined };
    case "FLDS":
    case "FLD":
      return { type: "module", module: "technical", symbol: ticker || undefined };
    case "ECO":
      return { type: "module", module: "economic" };
    case "N":
      return { type: "module", module: "news", symbol: ticker || undefined };
    case "PORT":
      return { type: "module", module: "portfolio" };
    case "SCREEN":
      return { type: "module", module: "screening", symbol: ticker || undefined };
    case "AI":
      return { type: "ai", module: "ai", query: parts.slice(1).join(" ") || undefined };
    case "BACKTEST":
      return { type: "backtest", module: "quant", symbol: ticker || undefined };
    case "WORKSPACE":
      return { type: "workspace", symbol: ticker || undefined };
    default:
      return { type: "ai", module: "ai", query: raw.trim() };
  }
}
