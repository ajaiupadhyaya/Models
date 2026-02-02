import React, { useState } from "react";
import { useTerminal } from "../TerminalContext";
import { PrimaryInstrument, type IndicatorOverlay } from "./PrimaryInstrument";

const INDICATORS: { id: IndicatorOverlay; label: string }[] = [
  { id: "none", label: "None" },
  { id: "sma20", label: "SMA 20" },
  { id: "sma50", label: "SMA 50" },
  { id: "ema12", label: "EMA 12" },
  { id: "ema26", label: "EMA 26" },
  { id: "rsi", label: "RSI" },
  { id: "macd", label: "MACD" },
  { id: "bollinger", label: "BB" },
  { id: "atr", label: "ATR" },
];

export const TechnicalPanel: React.FC = () => {
  const { primarySymbol } = useTerminal();
  const [overlay, setOverlay] = useState<IndicatorOverlay>("none");

  return (
    <section className="panel panel-main">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div className="panel-title">Technical: {primarySymbol}</div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {INDICATORS.map((ind) => (
            <button
              key={ind.id}
              type="button"
              className="ai-button"
              style={{
                padding: "4px 8px",
                fontSize: 11,
                background: overlay === ind.id ? "var(--accent)" : "var(--bg-panel)",
                border: `1px solid ${overlay === ind.id ? "var(--accent)" : "var(--border)"}`,
                color: overlay === ind.id ? "#0a0a0a" : "var(--text)",
              }}
              onClick={() => setOverlay(ind.id)}
            >
              {ind.label}
            </button>
          ))}
        </div>
      </div>
      <PrimaryInstrument indicatorOverlay={overlay} />
    </section>
  );
};
