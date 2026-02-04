/**
 * D3 yield curve (maturity vs yield). Used in Economic panel.
 */
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { CHART_MARGIN_PRESETS, getChartMargin } from "./theme";

export interface YieldCurveProps {
  maturities: number[];
  yields: number[];
  date?: string;
  width?: number;
  height?: number;
  marginPreset?: keyof typeof CHART_MARGIN_PRESETS;
  title?: string;
  className?: string;
  style?: React.CSSProperties;
}

export const YieldCurve: React.FC<YieldCurveProps> = ({
  maturities,
  yields,
  date,
  width: widthProp,
  height = 160,
  marginPreset = "default",
  title = "US Treasury yield curve",
  className = "chart-root",
  style,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || maturities.length < 2 || yields.length < 2) return;
    const el = ref.current;
    const margin = getChartMargin(marginPreset);
    const width = Math.min(widthProp ?? el.clientWidth ?? 320, 400);
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const n = Math.min(maturities.length, yields.length);
    const m = maturities.slice(0, n);
    const y = yields.slice(0, n);

    d3.select(el).selectAll("*").remove();
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([Math.min(...m), Math.max(...m)]).range([0, innerWidth]);
    const yScale = d3
      .scaleLinear()
      .domain([Math.min(...y) - 0.1, Math.max(...y) + 0.1])
      .nice()
      .range([innerHeight, 0]);

    const lineData = m.map((mat, i) => ({ m: mat, y: y[i]! }));
    const line = d3.line<{ m: number; y: number }>().x((d) => xScale(d.m)).y((d) => yScale(d.y));
    g.append("path")
      .datum(lineData)
      .attr("fill", "none")
      .attr("stroke", "var(--accent)")
      .attr("stroke-width", 2)
      .attr("d", line);
    g.selectAll("circle")
      .data(lineData)
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.m))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 4)
      .attr("fill", "var(--accent)");
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("class", "axis axis-x")
      .call(d3.axisBottom(xScale).tickFormat((d) => d + "Y"));
    g.append("g")
      .attr("class", "axis axis-y")
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => d + "%"));
    g.append("text")
      .attr("x", 2)
      .attr("y", 8)
      .attr("fill", "var(--text-soft)")
      .attr("font-size", 10)
      .attr("font-family", "var(--font-mono)")
      .text(date ? `${title} (${date})` : title);
  }, [maturities, yields, date, widthProp, height, marginPreset, title]);

  if (maturities.length < 2 || yields.length < 2) return null;
  return (
    <div ref={ref} className={className} style={{ minHeight: height, ...style }} />
  );
};
