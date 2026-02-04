/**
 * D3 chart utilities: scales, axes, responsive container.
 */
import * as d3 from "d3";

export interface TimeSeriesPoint {
  date: Date;
  value: number;
}

/** Create time scale for x-axis. */
export function createTimeScale(
  data: TimeSeriesPoint[],
  range: [number, number]
): d3.ScaleTime<number, number> {
  const extent = d3.extent(data, (d) => d.date) as [Date, Date];
  return d3.scaleTime().domain(extent).range(range).nice();
}

/** Create linear scale for y-axis. */
export function createLinearScale(
  data: TimeSeriesPoint[],
  range: [number, number],
  padding = 0
): d3.ScaleLinear<number, number> {
  const extent = d3.extent(data, (d) => d.value) as [number, number];
  const [lo, hi] = extent;
  const span = (hi - lo) || 1;
  const domain: [number, number] = [
    lo - span * padding,
    hi + span * padding,
  ];
  return d3.scaleLinear().domain(domain).range(range).nice();
}

/** Create linear scale from explicit domain. */
export function createLinearScaleFromDomain(
  domain: [number, number],
  range: [number, number]
): d3.ScaleLinear<number, number> {
  return d3.scaleLinear().domain(domain).range(range).nice();
}

/** Attach bottom axis to a D3 selection. */
export function drawAxisBottom(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  scale: d3.ScaleTime<number, number> | d3.ScaleLinear<number, number>,
  ticks = 5,
  tickFormat?: (d: d3.NumberValue) => string
): void {
  let axis = d3.axisBottom(scale).ticks(ticks);
  if (tickFormat) axis = axis.tickFormat(tickFormat);
  g.attr("class", "axis axis-x").call(axis);
}

/** Attach left axis to a D3 selection. */
export function drawAxisLeft(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  scale: d3.ScaleLinear<number, number>,
  ticks = 5,
  tickFormat?: (d: d3.NumberValue) => string
): void {
  let axis = d3.axisLeft(scale).ticks(ticks);
  if (tickFormat) axis = axis.tickFormat(tickFormat);
  g.attr("class", "axis axis-y").call(axis);
}

/** Get container width/height; fallback for SSR or missing clientWidth. */
export function getContainerSize(
  el: HTMLElement | null,
  fallbackWidth = 400,
  fallbackHeight = 200
): { width: number; height: number } {
  if (!el) return { width: fallbackWidth, height: fallbackHeight };
  return {
    width: el.clientWidth || fallbackWidth,
    height: el.clientHeight || fallbackHeight,
  };
}

/** Subscribe to container resize and run render. Returns cleanup. */
export function useResize(
  el: HTMLElement | null,
  render: (width: number, height: number) => void
): () => void {
  if (!el) return () => {};
  const ro = new ResizeObserver(() => {
    const w = el.clientWidth || 400;
    const h = el.clientHeight || 200;
    render(w, h);
  });
  ro.observe(el);
  const w = el.clientWidth || 400;
  const h = el.clientHeight || 200;
  render(w, h);
  return () => ro.disconnect();
}
