/**
 * D3 bar chart for risk metrics, model comparison, sector distribution, etc.
 * Includes tooltip on hover and axis labels.
 */
import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { CHART_MARGIN_PRESETS, getChartMargin } from "./theme";

export interface BarChartDatum {
  label: string;
  value: number;
  color?: string;
}

export interface BarChartProps {
  data: BarChartDatum[];
  width?: number;
  height?: number;
  marginPreset?: keyof typeof CHART_MARGIN_PRESETS;
  title?: string;
  valueFormat?: (v: number) => string;
  horizontal?: boolean;
  className?: string;
  style?: React.CSSProperties;
  xAxisLabel?: string;
  yAxisLabel?: string;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  width: widthProp,
  height = 200,
  marginPreset = "default",
  title,
  valueFormat = (v) => String(v),
  horizontal = false,
  className = "chart-root",
  style,
  xAxisLabel,
  yAxisLabel,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const setTooltipRef = useRef(setTooltip);
  setTooltipRef.current = setTooltip;

  useEffect(() => {
    if (!ref.current || data.length === 0) return;
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

    const labels = data.map((d) => d.label);
    const values = data.map((d) => d.value);
    const maxVal = Math.max(...values, 0);
    const minVal = Math.min(...values, 0);
    const span = maxVal - minVal || 1;

    const showTooltip = (evt: MouseEvent, d: BarChartDatum) => {
      const container = el.parentElement?.getBoundingClientRect() ?? el.getBoundingClientRect();
      setTooltipRef.current({
        x: evt.clientX - container.left,
        y: evt.clientY - container.top - 8,
        text: `${d.label}: ${valueFormat(d.value)}`,
      });
    };
    const hideTooltip = () => setTooltipRef.current(null);

    if (horizontal) {
      const yScale = d3.scaleBand().domain(labels).range([0, innerHeight]).padding(0.2);
      const xScale = d3.scaleLinear().domain([Math.min(0, minVal), Math.max(0, maxVal)]).range([0, innerWidth]);
      g.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yScale));
      g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .attr("class", "axis axis-x")
        .call(d3.axisBottom(xScale).ticks(5).tickFormat((v) => valueFormat(Number(v))));
      g.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("y", (d) => yScale(d.label) ?? 0)
        .attr("height", yScale.bandwidth())
        .attr("x", (d) => (d.value >= 0 ? xScale(0) : xScale(d.value)))
        .attr("width", (d) => Math.abs(xScale(d.value) - xScale(0)))
        .attr("fill", (d) => d.color ?? "var(--accent)")
        .attr("opacity", 0.85)
        .style("cursor", "pointer")
        .on("mouseover", showTooltip)
        .on("mousemove", showTooltip)
        .on("mouseleave", hideTooltip);
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
    } else {
      const xScale = d3.scaleBand().domain(labels).range([0, innerWidth]).padding(0.2);
      const yScale = d3.scaleLinear().domain([Math.min(0, minVal), Math.max(0, maxVal) + span * 0.1]).range([innerHeight, 0]);
      g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .attr("class", "axis axis-x")
        .call(d3.axisBottom(xScale));
      g.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yScale).ticks(5).tickFormat((v) => valueFormat(Number(v))));
      g.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (d) => xScale(d.label) ?? 0)
        .attr("width", xScale.bandwidth())
        .attr("y", (d) => (d.value >= 0 ? yScale(d.value) : yScale(0)))
        .attr("height", (d) => Math.abs(yScale(d.value) - yScale(0)))
        .attr("fill", (d) => d.color ?? "var(--accent)")
        .attr("opacity", 0.85)
        .style("cursor", "pointer")
        .on("mouseover", showTooltip)
        .on("mousemove", showTooltip)
        .on("mouseleave", hideTooltip);
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
  }, [data, widthProp, height, marginPreset, title, valueFormat, horizontal, xAxisLabel, yAxisLabel]);

  if (data.length === 0) return null;
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
