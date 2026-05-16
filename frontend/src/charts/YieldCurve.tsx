/**
 * D3 yield curve (maturity vs yield). Used in Economic panel.
 * Includes tooltips on data points and axis labels.
 */
import React, { useEffect, useRef, useState } from "react";
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
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const setTooltipRef = useRef(setTooltip);
  setTooltipRef.current = setTooltip;

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
    const circles = g
      .selectAll("circle")
      .data(lineData)
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.m))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 5)
      .attr("fill", "var(--accent)")
      .style("cursor", "pointer");
    circles.on("mouseover", function (evt, d) {
      setTooltipRef.current({
        x: xScale(d.m) + margin.left,
        y: yScale(d.y) + margin.top,
        text: `${d.m}Y: ${d.y.toFixed(2)}%`,
      });
    });
    circles.on("mouseleave", () => setTooltipRef.current(null));
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("class", "axis axis-x")
      .call(d3.axisBottom(xScale).tickFormat((d) => d + "Y"));
    g.append("g")
      .attr("class", "axis axis-y")
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => d + "%"));
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + margin.bottom - 4)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--text-soft)")
      .attr("font-size", 9)
      .attr("font-family", "var(--font-mono)")
      .text("Maturity");
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -margin.left + 12)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--text-soft)")
      .attr("font-size", 9)
      .attr("font-family", "var(--font-mono)")
      .text("Yield (%)");
    g.append("text")
      .attr("x", 2)
      .attr("y", 8)
      .attr("fill", "var(--text-soft)")
      .attr("font-size", 10)
      .attr("font-family", "var(--font-mono)")
      .text(date ? `${title} (${date})` : title);

    return () => setTooltipRef.current(null);
  }, [maturities, yields, date, widthProp, height, marginPreset, title]);

  if (maturities.length < 2 || yields.length < 2) return null;
  return (
    <div className={className} style={{ position: "relative", minHeight: height, ...style }}>
      <div ref={ref} style={{ width: "100%", minHeight: height }} />
      {tooltip && (
        <div className="chart-tooltip" style={{ left: tooltip.x, top: tooltip.y, transform: "translate(-50%, -100%)" }}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
};
