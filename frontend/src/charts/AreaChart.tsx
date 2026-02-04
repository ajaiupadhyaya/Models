/**
 * D3 area chart for equity curve, cumulative returns, etc.
 */
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { createTimeScale, createLinearScale, drawAxisBottom, drawAxisLeft, type TimeSeriesPoint } from "./utils";
import { CHART_MARGIN_PRESETS, getChartMargin } from "./theme";

export interface AreaChartProps {
  data: TimeSeriesPoint[];
  width?: number;
  height?: number;
  marginPreset?: keyof typeof CHART_MARGIN_PRESETS;
  title?: string;
  fill?: string;
  stroke?: string;
  className?: string;
  style?: React.CSSProperties;
}

export const AreaChart: React.FC<AreaChartProps> = ({
  data,
  width: widthProp,
  height = 180,
  marginPreset = "default",
  title,
  fill = "var(--accent)",
  stroke = "var(--accent)",
  className = "chart-root",
  style,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

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

    const xScale = createTimeScale(data, [0, innerWidth]);
    const yScale = createLinearScale(data, [innerHeight, 0], 0.05);

    const area = d3
      .area<TimeSeriesPoint>()
      .x((d) => xScale(d.date))
      .y0(innerHeight)
      .y1((d) => yScale(d.value));

    g.append("path").datum(data).attr("fill", fill).attr("stroke", "none").attr("opacity", 0.4).attr("d", area);
    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", stroke)
      .attr("stroke-width", 1.5)
      .attr(
        "d",
        d3
          .line<TimeSeriesPoint>()
          .x((d) => xScale(d.date))
          .y((d) => yScale(d.value))
      );

    const bottomG = g.append("g").attr("transform", `translate(0,${innerHeight})`);
    drawAxisBottom(bottomG as d3.Selection<SVGGElement, unknown, null, undefined>, xScale, 5);
    const leftG = g.append("g");
    drawAxisLeft(leftG as d3.Selection<SVGGElement, unknown, null, undefined>, yScale, 5);

    if (title) {
      g.append("text")
        .attr("x", 2)
        .attr("y", 8)
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 10)
        .attr("font-family", "var(--font-mono)")
        .text(title);
    }
  }, [data, widthProp, height, marginPreset, title, fill, stroke]);

  if (data.length < 2) return null;
  return (
    <div ref={ref} className={className} style={{ minHeight: height, ...style }} />
  );
};
