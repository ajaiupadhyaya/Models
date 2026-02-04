/**
 * D3 candlestick + volume chart component. Used by PrimaryInstrument (with overlays/zoom added there).
 */
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { getChartMargin } from "./theme";
import { drawCandles, drawVolumeBars, type Candle } from "./candlestickHelpers";

export interface CandlestickVolumeProps {
  data: Candle[];
  width?: number;
  priceHeight?: number;
  volumeHeight?: number;
  marginPreset?: "default" | "wide" | "compact";
  className?: string;
  style?: React.CSSProperties;
}

export const CandlestickVolume: React.FC<CandlestickVolumeProps> = ({
  data,
  width: widthProp,
  priceHeight = 260,
  volumeHeight = 80,
  marginPreset = "wide",
  className = "chart-root",
  style,
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || data.length === 0) return;
    const el = ref.current;
    const margin = getChartMargin(marginPreset);
    const width = widthProp ?? el.clientWidth ?? 600;
    const innerWidth = width - margin.left - margin.right;
    const innerPriceHeight = priceHeight - margin.top - 4;
    const innerVolumeHeight = volumeHeight - 4;
    const totalHeight = priceHeight + volumeHeight;

    d3.select(el).selectAll("*").remove();
    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", totalHeight);
    const marginG = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const priceG = marginG.append("g");
    const volumeG = marginG.append("g").attr("transform", `translate(0,${innerPriceHeight + 4})`);

    const xScale = d3
      .scaleTime()
      .domain(d3.extent(data, (d) => d.date) as [Date, Date])
      .range([0, innerWidth]);
    const yScale = d3
      .scaleLinear()
      .domain([
        (d3.min(data, (d) => d.low) ?? 0) * 0.998,
        (d3.max(data, (d) => d.high) ?? 1) * 1.002,
      ])
      .nice()
      .range([innerPriceHeight, 0]);
    const maxVol = d3.max(data, (d) => d.volume ?? 0) ?? 1;
    const yVolScale = d3.scaleLinear().domain([0, maxVol]).range([innerVolumeHeight, 0]).nice();

    priceG
      .append("g")
      .attr("transform", `translate(0,${innerPriceHeight})`)
      .attr("class", "axis axis-x")
      .call(d3.axisBottom(xScale).ticks(5));
    priceG.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yScale).ticks(5));

    drawCandles(priceG as d3.Selection<SVGGElement, unknown, null, undefined>, data, xScale, yScale, innerWidth);
    drawVolumeBars(
      volumeG as d3.Selection<SVGGElement, unknown, null, undefined>,
      data,
      xScale,
      yVolScale,
      innerWidth,
      innerVolumeHeight
    );
    volumeG
      .append("g")
      .attr("transform", `translate(0,${innerVolumeHeight})`)
      .attr("class", "axis axis-x axis-volume-x")
      .call(d3.axisBottom(xScale).ticks(3));
  }, [data, widthProp, priceHeight, volumeHeight, marginPreset]);

  if (data.length === 0) return null;
  return <div ref={ref} className={className} style={{ minHeight: priceHeight + volumeHeight, ...style }} />;
};
