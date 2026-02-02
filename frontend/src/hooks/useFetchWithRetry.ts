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

/** User-friendly messages for 429 and 5xx (used by panels). */
export function normalizeError(json: unknown, status: number): string {
  if (status === 429) {
    return "Too many requests. Try again in a minute.";
  }
  if (status >= 500 && status < 600) {
    return "Data temporarily unavailable. Please try again.";
  }
  if (json && typeof json === "object") {
    const d = (json as { detail?: unknown }).detail;
    if (d != null) return String(d);
    const e = (json as { error?: unknown }).error;
    if (e != null) return String(e);
  }
  return status >= 400 ? `HTTP ${status}` : "Request failed";
}

/** Retry-After header in seconds (for optional countdown UI). */
export function getRetryAfterSeconds(response: Response): number | null {
  const h = response.headers.get("Retry-After");
  if (h == null) return null;
  const n = parseInt(h, 10);
  return Number.isFinite(n) ? n : null;
}

/** Optional client-side cache: last successful result per url, short TTL (30s). */
const clientCache = new Map<string, { data: unknown; expiry: number }>();
const CLIENT_CACHE_TTL_MS = 30_000;

/** Clear client cache (e.g. for tests). */
export function clearFetchCache(): void {
  clientCache.clear();
}

export function useFetchWithRetry<T = unknown>(
  url: string | null,
  options: UseFetchWithRetryOptions<T> = {}
): {
  data: T | null;
  error: string | null;
  loading: boolean;
  retry: () => void;
  retryAfterSeconds: number | null;
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
  const [retryAfterSeconds, setRetryAfterSeconds] = useState<number | null>(null);

  const runFetch = useCallback(async () => {
    if (!url) {
      setLoading(false);
      setData(null);
      setError(null);
      setRetryAfterSeconds(null);
      return;
    }

    const now = Date.now();
    const cached = clientCache.get(url);
    if (cached && cached.expiry > now) {
      setData(cached.data as T);
      setError(null);
      setLoading(false);
      setRetryAfterSeconds(null);
      return;
    }

    setLoading(true);
    setError(null);
    setRetryAfterSeconds(null);

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
            if (res.status === 429) {
              const ra = getRetryAfterSeconds(res);
              setRetryAfterSeconds(ra);
            }
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
          const result = (parsed ?? json) as T;
          setData(result);
          setError(null);
          setRetryAfterSeconds(null);
          clientCache.set(url, { data: result, expiry: Date.now() + CLIENT_CACHE_TTL_MS });
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
    if (url) clientCache.delete(url);
    setRetryKey((k) => k + 1);
  }, [url]);

  return { data, error, loading, retry, retryAfterSeconds };
}
