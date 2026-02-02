import React, { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import { resolveApiUrl } from "../../apiBase";
import { useTerminal } from "../TerminalContext";
import { getAuthHeaders } from "../../hooks/useFetchWithRetry";

interface Candle {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

const TIMEFRAMES = [
  { label: "1D", period: "1d" },
  { label: "5D", period: "5d" },
  { label: "1M", period: "1mo" },
  { label: "3M", period: "3mo" },
] as const;

export type IndicatorOverlay = "none" | "sma20" | "sma50" | "ema12" | "ema26" | "rsi" | "macd" | "bollinger" | "atr";

function ema(data: Candle[], period: number): { date: Date; value: number }[] {
  const out: { date: Date; value: number }[] = [];
  const k = 2 / (period + 1);
  let prevEma = data[0]!.close;
  out.push({ date: data[0]!.date, value: prevEma });
  for (let i = 1; i < data.length; i++) {
    const emaVal = (data[i]!.close - prevEma) * k + prevEma;
    prevEma = emaVal;
    out.push({ date: data[i]!.date, value: emaVal });
  }
  return out;
}

function sma(data: Candle[], period: number): { date: Date; value: number }[] {
  const out: { date: Date; value: number }[] = [];
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += data[j]!.close;
    out.push({ date: data[i]!.date, value: sum / period });
  }
  return out;
}

function rsi(data: Candle[], period: number = 14): { date: Date; value: number }[] {
  const out: { date: Date; value: number }[] = [];
  for (let i = period; i < data.length; i++) {
    let gainSum = 0;
    let lossSum = 0;
    for (let j = Math.max(1, i - period + 1); j <= i; j++) {
      const change = data[j]!.close - data[j - 1]!.close;
      if (change > 0) gainSum += change;
      else lossSum += -change;
    }
    const avgGain = gainSum / period;
    const avgLoss = lossSum / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const value = 100 - 100 / (1 + rs);
    out.push({ date: data[i]!.date, value: Math.min(100, Math.max(0, value)) });
  }
  return out;
}

interface MacdPoint {
  date: Date;
  macd: number;
  signal: number;
  histogram: number;
}

function macd(data: Candle[], fast: number = 12, slow: number = 26, signalPeriod: number = 9): MacdPoint[] {
  const emaFast = ema(data, fast);
  const emaSlow = ema(data, slow);
  const macdLine: { date: Date; value: number }[] = [];
  for (let i = slow - 1; i < data.length; i++) {
    const fastVal = emaFast[i]?.value ?? data[i]!.close;
    const slowVal = emaSlow[i]?.value ?? data[i]!.close;
    macdLine.push({ date: data[i]!.date, value: fastVal - slowVal });
  }
  if (macdLine.length < signalPeriod) return [];
  const out: MacdPoint[] = [];
  const k = 2 / (signalPeriod + 1);
  let prevSignal = macdLine[0]!.value;
  out.push({
    date: macdLine[0]!.date,
    macd: macdLine[0]!.value,
    signal: prevSignal,
    histogram: macdLine[0]!.value - prevSignal,
  });
  for (let i = 1; i < macdLine.length; i++) {
    const sig = (macdLine[i]!.value - prevSignal) * k + prevSignal;
    prevSignal = sig;
    out.push({
      date: macdLine[i]!.date,
      macd: macdLine[i]!.value,
      signal: sig,
      histogram: macdLine[i]!.value - sig,
    });
  }
  return out;
}

interface BollingerPoint {
  date: Date;
  upper: number;
  middle: number;
  lower: number;
}

function bollinger(data: Candle[], period: number = 20, k: number = 2): BollingerPoint[] {
  const out: BollingerPoint[] = [];
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += data[j]!.close;
    const middle = sum / period;
    let varSum = 0;
    for (let j = i - period + 1; j <= i; j++) varSum += (data[j]!.close - middle) ** 2;
    const std = Math.sqrt(varSum / period) || 0;
    out.push({
      date: data[i]!.date,
      upper: middle + k * std,
      middle,
      lower: middle - k * std,
    });
  }
  return out;
}

function atr(data: Candle[], period: number = 14): { date: Date; value: number }[] {
  const out: { date: Date; value: number }[] = [];
  if (data.length < period) return out;
  const tr: number[] = [];
  for (let i = 0; i < data.length; i++) {
    const prev = data[i - 1];
    const high = data[i]!.high;
    const low = data[i]!.low;
    const prevClose = prev ? prev.close : data[i]!.close;
    const trVal = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    tr.push(trVal);
  }
  const k = 2 / (period + 1);
  let prevAtr = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  out.push({ date: data[period - 1]!.date, value: prevAtr });
  for (let i = period; i < data.length; i++) {
    prevAtr = (tr[i]! - prevAtr) * k + prevAtr;
    out.push({ date: data[i]!.date, value: prevAtr });
  }
  return out;
}

interface PrimaryInstrumentProps {
  indicatorOverlay?: IndicatorOverlay;
}

export const PrimaryInstrument: React.FC<PrimaryInstrumentProps> = ({ indicatorOverlay = "none" }) => {
  const { primarySymbol } = useTerminal();
  const [timeframe, setTimeframe] = useState<typeof TIMEFRAMES[number]["period"]>("3mo");
  const [data, setData] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [chartError, setChartError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [showSma, setShowSma] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const effectiveOverlay: IndicatorOverlay = indicatorOverlay !== "none" ? indicatorOverlay : (showSma ? "sma20" : "none");

  useEffect(() => {
    const fetchData = async () => {
      try {
        setChartError(null);
        const res = await fetch(resolveApiUrl(`/api/v1/backtest/sample-data?symbol=${primarySymbol}&period=${timeframe}`), { headers: getAuthHeaders() });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) {
          setChartError(json?.error ?? json?.detail ?? `HTTP ${res.status}`);
          setData([]);
          return;
        }
        const candles: Candle[] = (json.candles ?? []).map((c: { date: string; open: number; high: number; low: number; close: number; volume?: number }) => ({
          date: new Date(c.date),
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close),
          volume: c.volume != null ? Number(c.volume) : 0,
        }));
        setData(candles);
      } catch (err) {
        setChartError(err instanceof Error ? err.message : "Failed to load");
        setData([]);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [primarySymbol, timeframe, retryKey]);

  const sma20 = useMemo(() => (data.length >= 20 ? sma(data, 20) : []), [data]);
  const sma50 = useMemo(() => (data.length >= 50 ? sma(data, 50) : []), [data]);
  const ema12 = useMemo(() => (data.length >= 12 ? ema(data, 12) : []), [data]);
  const ema26 = useMemo(() => (data.length >= 26 ? ema(data, 26) : []), [data]);
  const rsi14 = useMemo(() => (data.length >= 15 ? rsi(data, 14) : []), [data]);
  const macdData = useMemo(() => (data.length >= 35 ? macd(data, 12, 26, 9) : []), [data]);
  const bollingerData = useMemo(() => (data.length >= 20 ? bollinger(data, 20, 2) : []), [data]);
  const atrData = useMemo(() => (data.length >= 15 ? atr(data, 14) : []), [data]);
  const smaOverlay =
    effectiveOverlay === "sma20" ? sma20
    : effectiveOverlay === "sma50" ? sma50
    : effectiveOverlay === "ema12" ? ema12
    : effectiveOverlay === "ema26" ? ema26
    : [];
  const showRsiPanel = effectiveOverlay === "rsi" && rsi14.length > 0;
  const showMacdPanel = effectiveOverlay === "macd" && macdData.length > 0;
  const showBollinger = effectiveOverlay === "bollinger" && bollingerData.length > 0;
  const showAtrPanel = effectiveOverlay === "atr" && atrData.length > 0;

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const tooltipEl = tooltipRef.current;

    const width = el.clientWidth || 600;
    const rsiPanelHeight = showRsiPanel ? 90 : 0;
    const macdPanelHeight = showMacdPanel ? 90 : 0;
    const atrPanelHeight = showAtrPanel ? 90 : 0;
    const totalHeight = 380 + rsiPanelHeight + macdPanelHeight + atrPanelHeight;
    const priceHeight = showRsiPanel || showMacdPanel || showAtrPanel ? 220 : 260;
    const volumeHeight = 80;
    const margin = { top: 24, right: 40, bottom: 28, left: 48 };
    const innerWidth = width - margin.left - margin.right;
    const innerPriceHeight = priceHeight - margin.top - 4;
    const innerVolumeHeight = volumeHeight - 4;

    d3.select(el).selectAll("*").remove();
    if (!data.length) {
      d3.select(el)
        .append("div")
        .attr("class", "panel-empty")
        .text(loading ? "Loading price data…" : (chartError ? "Price data unavailable. Try again." : "No data"));
      return;
    }

    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", totalHeight);

    const marginG = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const zoomable = marginG.append("g").attr("class", "zoomable");
    const priceG = zoomable.append("g");
    const volumeG = zoomable.append("g").attr("transform", `translate(0,${innerPriceHeight + 4})`);
    const rsiG = showRsiPanel ? zoomable.append("g").attr("transform", `translate(0,${innerPriceHeight + 4 + volumeHeight + 4})`) : null;
    const macdG = showMacdPanel ? zoomable.append("g").attr("transform", `translate(0,${innerPriceHeight + 4 + volumeHeight + 4 + (showRsiPanel ? 90 + 4 : 0)})`) : null;
    const atrG = showAtrPanel ? zoomable.append("g").attr("transform", `translate(0,${innerPriceHeight + 4 + volumeHeight + 4 + (showRsiPanel ? 90 + 4 : 0) + (showMacdPanel ? 90 + 4 : 0)})`) : null;

    const xScale = d3.scaleTime()
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

    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    priceG.append("g")
      .attr("transform", `translate(0,${innerPriceHeight})`)
      .attr("class", "axis axis-x")
      .call(d3.axisBottom(xScale).ticks(5));

    priceG.append("g").attr("class", "axis axis-y").call(yAxis);

    const candleWidth = Math.max(2, (innerWidth / data.length) * 0.6);
    const candlePad = (innerWidth / data.length - candleWidth) / 2;

    const candle = priceG
      .selectAll("g.candle")
      .data(data)
      .enter()
      .append("g")
      .attr("class", "candle")
      .attr("transform", (d, i) => {
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

    if (smaOverlay.length > 0) {
      const line = d3.line<{ date: Date; value: number }>()
        .x((d) => {
          const idx = data.findIndex((c) => c.date.getTime() === d.date.getTime());
          return ((idx >= 0 ? idx : data.length) / (data.length - 1 || 1)) * innerWidth;
        })
        .y((d) => yScale(d.value));
      priceG.append("path")
        .datum(smaOverlay)
        .attr("fill", "none")
        .attr("stroke", "var(--accent)")
        .attr("stroke-width", 2)
        .attr("d", line);
    }

    if (showBollinger && bollingerData.length > 0) {
      const lineX = (d: BollingerPoint) => {
        const idx = data.findIndex((c) => c.date.getTime() === d.date.getTime());
        return ((idx >= 0 ? idx : data.length) / (data.length - 1 || 1)) * innerWidth;
      };
      ["upper", "middle", "lower"].forEach((key, i) => {
        const line = d3.line<BollingerPoint>()
          .x(lineX)
          .y((d) => yScale(key === "upper" ? d.upper : key === "middle" ? d.middle : d.lower));
        priceG.append("path")
          .datum(bollingerData)
          .attr("fill", "none")
          .attr("stroke", i === 1 ? "var(--accent)" : "var(--text-soft)")
          .attr("stroke-width", i === 1 ? 1.5 : 1)
          .attr("stroke-dasharray", i === 1 ? "0" : "4,2")
          .attr("d", line);
      });
    }

    if (showRsiPanel && rsiG && rsi14.length > 0) {
      const rsiHeight = 80;
      const yRsi = d3.scaleLinear().domain([0, 100]).range([rsiHeight, 0]).nice();
      const lineRsi = d3.line<{ date: Date; value: number }>()
        .x((d) => {
          const idx = data.findIndex((c) => c.date.getTime() === d.date.getTime());
          return ((idx >= 0 ? idx : data.length) / (data.length - 1 || 1)) * innerWidth;
        })
        .y((d) => yRsi(d.value));
      rsiG.append("path")
        .datum(rsi14)
        .attr("fill", "none")
        .attr("stroke", "var(--accent)")
        .attr("stroke-width", 1.5)
        .attr("d", lineRsi);
      rsiG.append("line")
        .attr("x1", 0)
        .attr("x2", innerWidth)
        .attr("y1", yRsi(30))
        .attr("y2", yRsi(30))
        .attr("stroke", "var(--accent-red)")
        .attr("stroke-dasharray", "2,2")
        .attr("opacity", 0.6);
      rsiG.append("line")
        .attr("x1", 0)
        .attr("x2", innerWidth)
        .attr("y1", yRsi(70))
        .attr("y2", yRsi(70))
        .attr("stroke", "var(--accent-green)")
        .attr("stroke-dasharray", "2,2")
        .attr("opacity", 0.6);
      rsiG.append("g")
        .attr("class", "axis axis-y")
        .call(d3.axisLeft(yRsi).ticks(4));
      rsiG.append("text")
        .attr("x", 4)
        .attr("y", 10)
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 10)
        .attr("font-family", "var(--font-mono)")
        .text("RSI(14)");
    }

    if (showMacdPanel && macdG && macdData.length > 0) {
      const macdHeight = 80;
      const macdExtent = d3.extent(macdData, (d) => d.macd) as [number, number];
      const sigExtent = d3.extent(macdData, (d) => d.signal) as [number, number];
      const lo = Math.min(macdExtent[0], sigExtent[0]);
      const hi = Math.max(macdExtent[1], sigExtent[1]);
      const pad = (hi - lo) * 0.1 || 0.01;
      const yMacd = d3.scaleLinear().domain([lo - pad, hi + pad]).range([macdHeight, 0]).nice();
      const zeroY = yMacd(0);
      const lineX = (d: MacdPoint) => {
        const idx = data.findIndex((c) => c.date.getTime() === d.date.getTime());
        return ((idx >= 0 ? idx : data.length) / (data.length - 1 || 1)) * innerWidth;
      };
      macdData.forEach((d) => {
        const x = lineX(d);
        const yHist = yMacd(d.histogram);
        macdG.append("rect")
          .attr("x", x - (innerWidth / data.length) * 0.35)
          .attr("y", Math.min(zeroY, yHist))
          .attr("width", Math.max(1, (innerWidth / data.length) * 0.7))
          .attr("height", Math.max(1, Math.abs(yHist - zeroY)))
          .attr("fill", d.histogram >= 0 ? "var(--accent-green)" : "var(--accent-red)")
          .attr("opacity", 0.6);
      });
      macdG.append("path")
        .datum(macdData)
        .attr("fill", "none")
        .attr("stroke", "var(--accent)")
        .attr("stroke-width", 1.5)
        .attr("d", d3.line<MacdPoint>().x(lineX).y((d) => yMacd(d.macd)));
      macdG.append("path")
        .datum(macdData)
        .attr("fill", "none")
        .attr("stroke", "var(--accent-green)")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "2,2")
        .attr("d", d3.line<MacdPoint>().x(lineX).y((d) => yMacd(d.signal)));
      macdG.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yMacd).ticks(3));
      macdG.append("text")
        .attr("x", 4)
        .attr("y", 10)
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 10)
        .attr("font-family", "var(--font-mono)")
        .text("MACD(12,26,9)");
    }

    if (showAtrPanel && atrG && atrData.length > 0) {
      const atrHeight = 80;
      const atrExtent = d3.extent(atrData, (d) => d.value) as [number, number];
      const yAtr = d3.scaleLinear().domain(atrExtent).nice().range([atrHeight, 0]);
      const lineAtr = d3.line<{ date: Date; value: number }>()
        .x((d) => {
          const idx = data.findIndex((c) => c.date.getTime() === d.date.getTime());
          return ((idx >= 0 ? idx : data.length) / (data.length - 1 || 1)) * innerWidth;
        })
        .y((d) => yAtr(d.value));
      atrG.append("path")
        .datum(atrData)
        .attr("fill", "none")
        .attr("stroke", "var(--accent)")
        .attr("stroke-width", 1.5)
        .attr("d", lineAtr);
      atrG.append("g").attr("class", "axis axis-y").call(d3.axisLeft(yAtr).ticks(4));
      atrG.append("text")
        .attr("x", 4)
        .attr("y", 10)
        .attr("fill", "var(--text-soft)")
        .attr("font-size", 10)
        .attr("font-family", "var(--font-mono)")
        .text("ATR(14)");
    }

    data.forEach((d, i) => {
      const vol = d.volume ?? 0;
      const x = (i / (data.length - 1 || 1)) * innerWidth;
      volumeG
        .append("rect")
        .attr("x", x - (innerWidth / data.length) * 0.4)
        .attr("y", yVolScale(vol))
        .attr("width", Math.max(1, (innerWidth / data.length) * 0.8))
        .attr("height", Math.max(0, innerVolumeHeight - yVolScale(vol)))
        .attr("fill", d.close >= d.open ? "var(--accent-green)" : "var(--accent-red)")
        .attr("opacity", 0.7);
    });

    volumeG
      .append("g")
      .attr("transform", `translate(0,${innerVolumeHeight})`)
      .attr("class", "axis axis-x axis-volume-x")
      .call(d3.axisBottom(xScale).ticks(3));

    const bisect = d3.bisector((d: Candle) => d.date).left;
    const focus = priceG.append("g").attr("class", "focus").style("display", "none");
    focus.append("line").attr("class", "focus-line").attr("stroke", "var(--accent)").attr("stroke-dasharray", "2,2");
    focus.append("circle").attr("r", 4).attr("fill", "var(--accent)");

    const overlay = priceG
      .append("rect")
      .attr("width", innerWidth)
      .attr("height", innerPriceHeight)
      .attr("fill", "none")
      .attr("pointer-events", "all");

    overlay
      .on("mouseover", () => focus.style("display", "block"))
      .on("mouseout", () => {
        focus.style("display", "none");
        if (tooltipEl) tooltipEl.style.visibility = "hidden";
      })
      .on("mousemove", (event: MouseEvent) => {
        const x0 = xScale.invert(d3.pointer(event, el)[0] - margin.left);
        const i = Math.min(bisect(data, x0), data.length - 1);
        const d = data[i];
        if (!d) return;
        const xPos = (i / (data.length - 1 || 1)) * innerWidth;
        focus.attr("transform", `translate(${xPos},${yScale(d.close)})`);
        focus.select(".focus-line")
          .attr("x1", xPos)
          .attr("x2", xPos)
          .attr("y1", yScale(d.close))
          .attr("y2", innerPriceHeight);
        if (tooltipEl) {
          tooltipEl.style.visibility = "visible";
          tooltipEl.style.left = `${event.pageX + 12}px`;
          tooltipEl.style.top = `${event.pageY + 12}px`;
          tooltipEl.innerHTML = `
            <div class="num-mono" style="font-size:11px;color:var(--accent);margin-bottom:4px">${d.date.toLocaleDateString()}</div>
            <div class="num-mono" style="font-size:11px">O ${d.open.toFixed(2)} H ${d.high.toFixed(2)} L ${d.low.toFixed(2)} C ${d.close.toFixed(2)}</div>
            ${(d.volume ?? 0) > 0 ? `<div class="num-mono" style="font-size:11px;color:var(--text-soft)">Vol ${(d.volume ?? 0).toLocaleString()}</div>` : ""}
          `;
        }
      });

    const zoom = d3.zoom<SVGGElement, unknown>()
      .scaleExtent([1, 32])
      .on("zoom", (event) => {
        const t = event.transform;
        const tXOnly = d3.zoomIdentity.translate(t.x, 0).scale(t.k, 1);
        zoomable.attr("transform", tXOnly.toString());
      });

    zoomable.call(zoom as unknown as (selection: d3.Selection<SVGGElement, unknown, null, undefined>) => void);
  }, [data, loading, chartError, smaOverlay, showRsiPanel, rsi14, showBollinger, bollingerData, showMacdPanel, macdData, showAtrPanel, atrData]);

  const inlineCssVarsInSvgString = (svgString: string): string => {
    const root = document.documentElement;
    const getVar = (name: string) => getComputedStyle(root).getPropertyValue(name).trim() || name;
    const vars: Record<string, string> = {
      "var(--accent)": getVar("--accent"),
      "var(--text)": getVar("--text"),
      "var(--text-soft)": getVar("--text-soft"),
      "var(--accent-green)": getVar("--accent-green"),
      "var(--accent-red)": getVar("--accent-red"),
      "var(--bg-panel)": getVar("--bg-panel"),
      "var(--border)": getVar("--border"),
      "var(--font-mono)": getVar("--font-mono"),
    };
    let out = svgString;
    for (const [k, v] of Object.entries(vars)) {
      out = out.split(k).join(v || k);
    }
    return out;
  };

  const handleExportChart = (format: "svg" | "png" = "svg") => {
    const svgEl = ref.current?.querySelector("svg");
    if (!svgEl) return;
    const serializer = new XMLSerializer();
    let str = serializer.serializeToString(svgEl);
    str = inlineCssVarsInSvgString(str);
    if (format === "svg") {
      const blob = new Blob([str], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chart-${primarySymbol}-${timeframe}.svg`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }
    const img = new Image();
    const blob = new Blob([str], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      const w = img.naturalWidth || 600;
      const h = img.naturalHeight || 400;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        return;
      }
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--bg-panel").trim() || "#1a1a1a";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((b) => {
        if (!b) {
          URL.revokeObjectURL(url);
          return;
        }
        const a = document.createElement("a");
        a.href = URL.createObjectURL(b);
        a.download = `chart-${primarySymbol}-${timeframe}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
        URL.revokeObjectURL(url);
      }, "image/png");
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  };

  return (
    <section className="panel panel-main">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div className="panel-title">Primary Instrument: {primarySymbol}</div>
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          {data.length > 0 && (
            <>
              <button type="button" className="ai-button" style={{ padding: "4px 8px", fontSize: 11 }} onClick={() => handleExportChart("png")} title="Download chart as PNG">
                Export PNG
              </button>
              <button type="button" className="ai-button" style={{ padding: "4px 8px", fontSize: 11 }} onClick={() => handleExportChart("svg")} title="Download chart as SVG">
                Export SVG
              </button>
            </>
          )}
          {indicatorOverlay === "none" && (
            <button
              type="button"
              className="ai-button"
              style={{
                padding: "4px 8px",
                fontSize: 11,
                background: showSma ? "var(--accent)" : "var(--bg-panel)",
                border: `1px solid ${showSma ? "var(--accent)" : "var(--border)"}`,
                color: showSma ? "#0a0a0a" : "var(--text)",
              }}
              onClick={() => setShowSma(!showSma)}
            >
              SMA 20
            </button>
          )}
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.period}
              type="button"
              className="ai-button"
              style={{
                padding: "4px 8px",
                fontSize: 11,
                background: timeframe === tf.period ? "var(--accent)" : "var(--bg-panel)",
                border: `1px solid ${timeframe === tf.period ? "var(--accent)" : "var(--border)"}`,
                color: timeframe === tf.period ? "#0a0a0a" : "var(--text)",
              }}
              onClick={() => setTimeframe(tf.period)}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>
      {chartError && (
        <div className="panel-body-muted" style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
          {chartError}. Try again or retry.
          <button type="button" className="ai-button" onClick={() => setRetryKey((k) => k + 1)}>Retry</button>
        </div>
      )}
      <div style={{ position: "relative" }}>
        <div ref={ref} className="chart-root" />
        <div
          ref={tooltipRef}
          style={{
            position: "fixed",
            visibility: "hidden",
            pointerEvents: "none",
            background: "var(--bg-panel)",
            border: "1px solid var(--border)",
            borderRadius: 2,
            padding: "8px 10px",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            zIndex: 1000,
          }}
        />
      </div>
    </section>
  );
};
