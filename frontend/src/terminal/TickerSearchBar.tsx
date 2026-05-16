import React, { useState, useCallback, useRef, useEffect } from "react";
import { resolveApiUrl } from "../apiBase";
import { useTerminal } from "./TerminalContext";

interface SearchResult {
  symbol: string;
  name?: string;
  sector?: string;
  industry?: string;
  market_cap?: number | null;
}

export const TickerSearchBar: React.FC = () => {
  const { setPrimarySymbol, setActiveModule } = useTerminal();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const doSearch = useCallback(async (q: string) => {
    if (!q || q.length < 1) {
      setResults([]);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(resolveApiUrl(`/api/v1/equity/search?q=${encodeURIComponent(q)}`));
      const json = await res.json().catch(() => ({}));
      const items = (json.results ?? json) as SearchResult[];
      setResults(Array.isArray(items) ? items.slice(0, 10) : []);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const q = query.trim();
    if (q.length < 1) {
      setResults([]);
      return;
    }
    debounceRef.current = setTimeout(() => doSearch(q), 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, doSearch]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectTicker = useCallback(
    (symbol: string) => {
      setPrimarySymbol(symbol);
      setActiveModule("fundamental");
      setQuery("");
      setResults([]);
      setOpen(false);
    },
    [setPrimarySymbol, setActiveModule]
  );

  return (
    <div ref={containerRef} style={{ position: "relative", width: 220 }}>
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        placeholder="Search ticker..."
        style={{
          width: "100%",
          background: "var(--bg-panel)",
          border: "1px solid var(--border)",
          color: "var(--text)",
          padding: "6px 12px",
          borderRadius: 4,
          fontSize: 12,
          fontFamily: "var(--font-mono)",
        }}
        aria-label="Ticker search"
      />
      {open && (query.trim().length > 0 || results.length > 0) && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            marginTop: 4,
            background: "var(--bg-panel)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            boxShadow: "var(--shadow-elevated)",
            zIndex: 50,
            maxHeight: 240,
            overflowY: "auto",
          }}
          role="listbox"
        >
          {loading ? (
            <div style={{ padding: 12, color: "var(--text-soft)", fontSize: 12 }}>Searching...</div>
          ) : results.length === 0 ? (
            <div style={{ padding: 12, color: "var(--text-soft)", fontSize: 12 }}>
              {query.trim().length >= 1 ? "No results" : "Enter a ticker or company name"}
            </div>
          ) : (
            results.map((r) => (
              <button
                key={r.symbol}
                type="button"
                style={{
                  width: "100%",
                  textAlign: "left",
                  padding: "8px 12px",
                  background: "transparent",
                  border: "none",
                  color: "var(--text)",
                  fontSize: 12,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.04)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                }}
                onClick={() => selectTicker(r.symbol)}
                role="option"
              >
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent)" }}>{r.symbol}</span>
                <span style={{ color: "var(--text-soft)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginLeft: 8, maxWidth: 140 }}>
                  {r.name ?? ""}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
};
