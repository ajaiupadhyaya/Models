/**
 * D3 area chart for equity curve, cumulative returns, etc.
 * Includes tooltip, axis labels, and value formatting.
 */
import React, { useEffect, useRef, useState, useCallback } from "react";
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
  /** Format y-axis and tooltip values (default: 2 decimals) */
  valueFormat?: (v: number) => string;
  xAxisLabel?: string;
  yAxisLabel?: string;
}

const formatDate = (d: Date) => d3.timeFormat("%Y-%m-%d")(d);

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
  valueFormat = (v) => v.toFixed(2),
  xAxisLabel,
  yAxisLabel,
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
    drawAxisLeft(leftG as d3.Selection<SVGGElement, unknown, null, undefined>, yScale, 5, (d) => valueFormat(Number(d)));

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

    const bisect = d3.bisector((d: TimeSeriesPoint) => d.date).left;
    const overlay = g
      .append("rect")
      .attr("class", "chart-overlay")
      .attr("width", innerWidth)
      .attr("height", innerHeight);
    overlay.on("mousemove", function (evt) {
      const [mx] = d3.pointer(evt, this);
      const x0 = xScale.invert(mx);
      const i = Math.min(bisect(data, x0), data.length - 1);
      const d = data[i]!;
      setTooltipRef.current({
        x: mx + margin.left,
        y: margin.top + 12,
        text: `${formatDate(d.date)}: ${valueFormat(d.value)}`,
      });
    });
    overlay.on("mouseleave", () => setTooltipRef.current(null));

    return () => setTooltipRef.current(null);
  }, [data, widthProp, height, marginPreset, title, fill, stroke, valueFormat, xAxisLabel, yAxisLabel]);

  if (data.length < 2) return null;
  return (
    <div className={className} style={{ position: "relative", minHeight: height, ...style }}>
      <div ref={ref} style={{ width: "100%", minHeight: height }} />
      {tooltip && (
        <div className="chart-tooltip" style={{ left: tooltip.x, top: tooltip.y, transform: "translate(-50%, 0)" }}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
};
