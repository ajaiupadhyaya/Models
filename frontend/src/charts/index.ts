/**
 * Shared D3 chart library for the terminal.
 * All economic/finance charts in the terminal use these components and utilities.
 */
export { TimeSeriesLine } from "./TimeSeriesLine";
export type { TimeSeriesLineProps } from "./TimeSeriesLine";
export { BarChart } from "./BarChart";
export type { BarChartProps, BarChartDatum } from "./BarChart";
export { Heatmap } from "./Heatmap";
export type { HeatmapProps } from "./Heatmap";
export { YieldCurve } from "./YieldCurve";
export type { YieldCurveProps } from "./YieldCurve";
export { AreaChart } from "./AreaChart";
export type { AreaChartProps } from "./AreaChart";
export { CandlestickVolume } from "./CandlestickVolume";
export type { CandlestickVolumeProps } from "./CandlestickVolume";
export { drawCandles, drawVolumeBars } from "./candlestickHelpers";
export type { Candle } from "./candlestickHelpers";
export {
  CHART_MARGIN_PRESETS,
  getChartMargin,
  getComputedChartColors,
  inlineCssVarsInSvgString,
} from "./theme";
export type { ChartMarginPreset } from "./theme";
export {
  createTimeScale,
  createLinearScale,
  createLinearScaleFromDomain,
  drawAxisBottom,
  drawAxisLeft,
  getContainerSize,
  useResize,
} from "./utils";
export type { TimeSeriesPoint } from "./utils";
