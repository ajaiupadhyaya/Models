/**
 * API base URL for deployment when frontend and backend are on different origins.
 * Set VITE_API_ORIGIN in .env (e.g. https://your-api.onrender.com) so all /api/* requests
 * and WebSocket connections use that origin. Leave unset for same-origin (default).
 */
function getBase(): string {
  const env = typeof import.meta !== "undefined" && import.meta.env?.VITE_API_ORIGIN;
  const s = typeof env === "string" ? env.trim() : "";
  return s ? s.replace(/\/$/, "") : "";
}

export function getApiBase(): string {
  return getBase();
}

/** WebSocket origin (wss when API is https). Used when VITE_API_ORIGIN is set. */
export function getWsBase(): string {
  const base = getBase();
  if (!base) return "";
  try {
    const u = new URL(base);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.origin;
  } catch {
    return "";
  }
}

/** Resolve path to full URL when VITE_API_ORIGIN is set; otherwise return path for same-origin. */
export function resolveApiUrl(path: string): string {
  const base = getApiBase();
  if (!base || !path.startsWith("/")) return path;
  return base + path;
}
