/**
 * D3 correlation (or matrix) heatmap. Used for correlation matrix in Economic panel.
 */
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export interface HeatmapProps {
  symbols: string[];
  matrix: number[][];
  cellSize?: number;
  title?: string;
  domain?: [number, number];
  colorRange?: [string, string, string];
  className?: string;
  style?: React.CSSProperties;
}

export const Heatmap: React.FC<HeatmapProps> = ({
  symbols,
  matrix,
  cellSize = 28,
  title,
  domain = [-1, 0, 1],
  colorRange = ["#2166ac", "#333333", "#b2182b"],
  className = "chart-root",
  style,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || !symbols.length || !matrix.length) return;
    const el = ref.current;
    const n = symbols.length;
    const margin = { top: 8, left: 64 };
    const width = margin.left + n * cellSize;
    const height = margin.top + n * cellSize;

    d3.select(el).selectAll("*").remove();
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleLinear<string>().domain(domain).range(colorRange);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i]?.[j] ?? 0;
        g.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 2)
          .attr("height", cellSize - 2)
          .attr("fill", colorScale(v))
          .attr("stroke", "var(--border)")
          .attr("stroke-width", 0.5);
        g.append("text")
          .attr("x", j * cellSize + (cellSize - 2) / 2)
          .attr("y", i * cellSize + (cellSize - 2) / 2 + 4)
          .attr("text-anchor", "middle")
          .attr("font-size", 9)
          .attr("font-family", "var(--font-mono)")
          .attr("fill", Math.abs(v) > 0.5 ? "#fff" : "var(--text)")
          .text(typeof v === "number" ? v.toFixed(2) : "");
      }
      g.append("text")
        .attr("x", -6)
        .attr("y", i * cellSize + (cellSize - 2) / 2 + 4)
        .attr("text-anchor", "end")
        .attr("font-size", 9)
        .attr("font-family", "var(--font-mono)")
        .attr("fill", "var(--text-soft)")
        .text(symbols[i] ?? "");
    }
    for (let j = 0; j < n; j++) {
      g.append("text")
        .attr("x", j * cellSize + (cellSize - 2) / 2)
        .attr("y", -4)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("font-family", "var(--font-mono)")
        .attr("fill", "var(--text-soft)")
        .text(symbols[j] ?? "");
    }
    if (title) {
      svg
        .append("text")
        .attr("x", margin.left)
        .attr("y", 6)
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 10)
        .attr("font-family", "var(--font-mono)")
        .text(title);
    }
  }, [symbols, matrix, cellSize, title, domain, colorRange]);

  if (!symbols.length || !matrix.length) return null;
  return <div ref={ref} className={className} style={style} />;
};
