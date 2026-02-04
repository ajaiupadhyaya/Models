/**
 * D3 candlestick and volume drawing helpers. Used by PrimaryInstrument and CandlestickVolume.
 */
import * as d3 from "d3";

export interface Candle {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

/** Draw candlestick bodies and wicks on a D3 group. */
export function drawCandles(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: Candle[],
  xScale: d3.ScaleTime<number, number>,
  yScale: d3.ScaleLinear<number, number>,
  innerWidth: number,
  candleWidthRatio = 0.6
): void {
  const candleWidth = Math.max(2, (innerWidth / data.length) * candleWidthRatio);
  const candlePad = (innerWidth / data.length - candleWidth) / 2;
  const candle = g
    .selectAll("g.candle")
    .data(data)
    .enter()
    .append("g")
    .attr("class", "candle")
    .attr("transform", (_d, i) => {
      const x = (i / (data.length - 1 || 1)) * innerWidth;
      return `translate(${x},0)`;
    });
  candle
    .append("line")
    .attr("y1", (d) => yScale(d.high))
    .attr("y2", (d) => yScale(d.low))
    .attr("x1", innerWidth / data.length / 2)
    .attr("x2", innerWidth / data.length / 2)
    .attr("stroke", "var(--text-soft)")
    .attr("stroke-width", 1);
  candle
    .append("rect")
    .attr("y", (d) => yScale(Math.max(d.open, d.close)))
    .attr("height", (d) => Math.max(1, Math.abs(yScale(d.open) - yScale(d.close))))
    .attr("width", candleWidth)
    .attr("x", candlePad)
    .attr("fill", (d) => (d.close >= d.open ? "var(--accent-green)" : "var(--accent-red)"));
}

/** Draw volume bars (up/down colored) on a D3 group. */
export function drawVolumeBars(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: Candle[],
  xScale: d3.ScaleTime<number, number>,
  yVolScale: d3.ScaleLinear<number, number>,
  innerWidth: number,
  innerVolumeHeight: number
): void {
  data.forEach((d, i) => {
    const vol = d.volume ?? 0;
    const x = (i / (data.length - 1 || 1)) * innerWidth;
    g.append("rect")
      .attr("x", x - (innerWidth / data.length) * 0.4)
      .attr("y", yVolScale(vol))
      .attr("width", Math.max(1, (innerWidth / data.length) * 0.8))
      .attr("height", Math.max(0, innerVolumeHeight - yVolScale(vol)))
      .attr("fill", d.close >= d.open ? "var(--accent-green)" : "var(--accent-red)")
      .attr("opacity", 0.7);
  });
}
