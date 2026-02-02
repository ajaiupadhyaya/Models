import { useEffect, useState } from "react";
import { getWsBase } from "../apiBase";

interface PriceUpdate {
  type: string;
  symbol: string;
  price: number;
  timestamp?: string;
}

/**
 * Connect to backend WebSocket for live price updates.
 * Falls back to no data if WebSocket is unavailable (REST polling still used by parent).
 */
export function useWebSocketPrice(symbol: string | null): {
  price: number | null;
  connected: boolean;
  error: string | null;
} {
  const [price, setPrice] = useState<number | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) {
      setPrice(null);
      setConnected(false);
      setError(null);
      return;
    }
    const wsBase = getWsBase();
    const wsUrl = wsBase
      ? `${wsBase}/api/v1/ws/prices/${symbol}`
      : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/v1/ws/prices/${symbol}`;
    let ws: WebSocket | null = null;
    let closed = false;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
          if (!closed) setConnected(true);
        };
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as PriceUpdate;
            if (data.type === "price_update" && typeof data.price === "number") {
              setPrice(data.price);
            }
          } catch {
            // ignore parse errors
          }
        };
        ws.onerror = () => {
          if (!closed) setError("WebSocket error");
        };
        ws.onclose = () => {
          if (!closed) {
            setConnected(false);
          }
        };
      } catch (e) {
        setError(e instanceof Error ? e.message : "WebSocket failed");
      }
    };

    connect();
    return () => {
      closed = true;
      if (ws?.readyState === WebSocket.OPEN) ws.close();
      setConnected(false);
      setPrice(null);
      setError(null);
    };
  }, [symbol]);

  return { price, connected, error };
}
