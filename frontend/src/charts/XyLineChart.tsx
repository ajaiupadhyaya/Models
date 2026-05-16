/**
 * D3 XY line chart for scatter/line data (e.g. efficient frontier: volatility vs return).
 * Includes tooltip and axis labels.
 */
import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { CHART_MARGIN_PRESETS, getChartMargin } from "./theme";

export interface XyPoint {
  x: number;
  y: number;
  label?: string;
}

export interface XyLineChartProps {
  data: XyPoint[];
  width?: number;
  height?: number;
  marginPreset?: keyof typeof CHART_MARGIN_PRESETS;
  title?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  className?: string;
  style?: React.CSSProperties;
}

export const XyLineChart: React.FC<XyLineChartProps> = ({
  data,
  width: widthProp,
  height = 180,
  marginPreset = "default",
  title,
  xFormat = (v) => v.toFixed(2),
  yFormat = (v) => v.toFixed(2),
  xAxisLabel,
  yAxisLabel,
  className = "chart-root",
  style,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const setTooltipRef = useRef(setTooltip);
  setTooltipRef.current = setTooltip;

  useEffect(() => {
    if (!ref.current || data.length < 2) return;
    const el = ref.current;
    const margin = getChartMargin(marginPreset);
    const width = widthProp ?? el.clientWidth ?? 400;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(el).selectAll("*").remove();
    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(data, (d) => d.x) as [number, number];
    const yExtent = d3.extent(data, (d) => d.y) as [number, number];
    const xScale = d3.scaleLinear().domain(xExtent).range([0, innerWidth]).nice();
    const yScale = d3.scaleLinear().domain(yExtent).range([innerHeight, 0]).nice();

    const line = d3.line<XyPoint>().x((d) => xScale(d.x)).y((d) => yScale(d.y));
    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "var(--accent)")
      .attr("stroke-width", 1.5)
      .attr("d", line);
    g.selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 4)
      .attr("fill", "var(--accent)")
      .style("cursor", "pointer")
      .on("mouseover", function (evt, d) {
        setTooltipRef.current({
          x: xScale(d.x) + margin.left,
          y: yScale(d.y) + margin.top,
          text: d.label ?? `${xFormat(d.x)}, ${yFormat(d.y)}`,
        });
      })
      .on("mouseleave", () => setTooltipRef.current(null));

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("class", "axis axis-x")
      .call(d3.axisBottom(xScale).ticks(5).tickFormat((d) => xFormat(Number(d))));
    g.append("g")
      .attr("class", "axis axis-y")
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => yFormat(Number(d))));

    if (xAxisLabel) {
      g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight + margin.bottom - 4)
        .attr("text-anchor", "middle")
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 9)
        .attr("font-family", "var(--font-mono)")
        .text(xAxisLabel);
    }
    if (yAxisLabel) {
      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -margin.left + 12)
        .attr("text-anchor", "middle")
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 9)
        .attr("font-family", "var(--font-mono)")
        .text(yAxisLabel);
    }
    if (title) {
      g.append("text")
        .attr("x", 2)
        .attr("y", 8)
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 10)
        .attr("font-family", "var(--font-mono)")
        .text(title);
    }

    return () => setTooltipRef.current(null);
  }, [data, widthProp, height, marginPreset, title, xFormat, yFormat, xAxisLabel, yAxisLabel]);

  if (data.length < 2) return null;
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
