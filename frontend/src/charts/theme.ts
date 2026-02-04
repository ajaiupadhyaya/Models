/**
 * Chart theme and layout config driven by CSS variables.
 * Used by all D3 chart components in the terminal.
 */

export const CHART_MARGIN_PRESETS = {
  default: { top: 12, right: 12, bottom: 24, left: 44 },
  compact: { top: 8, right: 8, bottom: 20, left: 40 },
  wide: { top: 24, right: 40, bottom: 28, left: 48 },
  sparkline: { top: 2, right: 2, bottom: 2, left: 2 },
} as const;

export type ChartMarginPreset = keyof typeof CHART_MARGIN_PRESETS;

export function getChartMargin(preset: ChartMarginPreset = "default") {
  return { ...CHART_MARGIN_PRESETS[preset] };
}

/** Resolve CSS variable to computed color for export (SVG/PNG). */
export function getComputedChartColors(): Record<string, string> {
  if (typeof document === "undefined") {
    return {
      accent: "#e8a020",
      text: "#e8e8e8",
      textSoft: "#8a8a8a",
      accentGreen: "#22c55e",
      accentRed: "#ef4444",
      bgPanel: "#111111",
      border: "#262626",
      fontMono: "JetBrains Mono, monospace",
    };
  }
  const root = document.documentElement;
  const get = (name: string) => getComputedStyle(root).getPropertyValue(name).trim() || name;
  return {
    accent: get("--accent"),
    text: get("--text"),
    textSoft: get("--text-soft"),
    accentGreen: get("--accent-green"),
    accentRed: get("--accent-red"),
    bgPanel: get("--bg-panel"),
    border: get("--border"),
    fontMono: get("--font-mono"),
  };
}

/** Inline CSS vars in an SVG string for export (so SVG looks correct when saved). */
export function inlineCssVarsInSvgString(svgString: string): string {
  const colors = getComputedChartColors();
  const vars: Record<string, string> = {
    "var(--accent)": colors.accent,
    "var(--text)": colors.text,
    "var(--text-soft)": colors.textSoft,
    "var(--accent-green)": colors.accentGreen,
    "var(--accent-red)": colors.accentRed,
    "var(--bg-panel)": colors.bgPanel,
    "var(--border)": colors.border,
    "var(--font-mono)": colors.fontMono,
  };
  let out = svgString;
  for (const [k, v] of Object.entries(vars)) {
    out = out.split(k).join(v || k);
  }
  return out;
}
