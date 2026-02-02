import { useCallback, useEffect, useState } from "react";

export interface UseFetchWithRetryOptions<T> {
  /** Request init (method, headers, body). */
  requestInit?: RequestInit;
  /** Dependencies that trigger refetch when changed (default: [url]). */
  deps?: unknown[];
  /** Max retry attempts on failure (default: 2). */
  maxRetries?: number;
  /** Delay in ms before first retry; doubles each time (default: 1000). */
  retryDelayMs?: number;
  /** If provided, response is parsed and validated; return null to treat as error. */
  parse?: (json: unknown) => T | null;
  /** Retry only on 5xx and network errors (default: true). No retry on 4xx. */
  retryOn5xxOnly?: boolean;
}

function normalizeError(json: unknown, status: number): string {
  if (json && typeof json === "object") {
    const d = (json as { detail?: unknown }).detail;
    if (d != null) return String(d);
    const e = (json as { error?: unknown }).error;
    if (e != null) return String(e);
  }
  return status >= 400 ? `HTTP ${status}` : "Request failed";
}

export function useFetchWithRetry<T = unknown>(
  url: string | null,
  options: UseFetchWithRetryOptions<T> = {}
): {
  data: T | null;
  error: string | null;
  loading: boolean;
  retry: () => void;
} {
  const {
    requestInit,
    deps,
    maxRetries = 2,
    retryDelayMs = 1000,
    parse,
    retryOn5xxOnly = true,
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retryKey, setRetryKey] = useState(0);

  const runFetch = useCallback(async () => {
    if (!url) {
      setLoading(false);
      setData(null);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    let attempt = 0;

    const doFetch = (): Promise<void> =>
      fetch(url, requestInit)
        .then(async (res) => {
          const json = await res.json().catch(() => ({}));
          const errMsg = normalizeError(json, res.status);
          const is5xx = res.status >= 500 && res.status < 600;
          const shouldRetry =
            attempt < maxRetries &&
            (retryOn5xxOnly ? is5xx : !res.ok);

          if (!res.ok) {
            if (shouldRetry) {
              attempt += 1;
              const delay = retryDelayMs * Math.pow(2, attempt - 1);
              await new Promise((r) => setTimeout(r, delay));
              return doFetch();
            }
            setError(errMsg);
            setData(null);
            return;
          }

          const parsed = parse ? parse(json) : (json as T);
          if (parse !== undefined && parsed === null) {
            setError(normalizeError(json, res.status) || "Invalid response");
            setData(null);
            return;
          }
          setData((parsed ?? json) as T);
          setError(null);
        })
        .catch((err: unknown) => {
          const message = err instanceof Error ? err.message : "Network error";
          if (attempt < maxRetries) {
            attempt += 1;
            const delay = retryDelayMs * Math.pow(2, attempt - 1);
            return new Promise<void>((r) => setTimeout(r, delay)).then(doFetch);
          }
          setError(message);
          setData(null);
        })
        .finally(() => {
          setLoading(false);
        });

    doFetch();
  }, [url, maxRetries, retryDelayMs, parse, requestInit, retryOn5xxOnly, retryKey]);

  const effectiveDeps = deps ?? [url];
  useEffect(() => {
    runFetch();
  }, [...effectiveDeps, retryKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const retry = useCallback(() => {
    setRetryKey((k) => k + 1);
  }, []);

  return { data, error, loading, retry };
}
