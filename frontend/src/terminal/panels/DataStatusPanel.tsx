import React, { useEffect, useState, useCallback } from "react";
import { resolveApiUrl } from "../../apiBase";

interface DataStatusItem {
  source: string;
  entity: string;
  last_updated: string | null;
  last_error: string | null;
}

function dataTypeForRow(row: DataStatusItem): string | null {
  if (row.source === "yfinance" && row.entity === "ohlcv") return "ohlcv";
  if (row.source === "fred" && row.entity === "macro") return "macro";
  if (row.source === "newsapi" && row.entity === "news") return "news";
  if (row.source === "fmp" && row.entity === "fundamentals") return "fundamentals";
  return null;
}

export const DataStatusPanel: React.FC = () => {
  const [items, setItems] = useState<DataStatusItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<Record<string, boolean>>({});

  const fetchStatus = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(resolveApiUrl("/api/v1/data/status"))
      .then((r) => r.json())
      .then((data: { items?: DataStatusItem[]; error?: string }) => {
        if (data.items) setItems(data.items);
        if (data.error) setError(data.error);
      })
      .catch((e) => setError(e.message || "Failed to load data status"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleRefresh = useCallback(
    async (dataType: string) => {
      setRefreshing((prev) => ({ ...prev, [dataType]: true }));
      try {
        const res = await fetch(resolveApiUrl(`/api/v1/data/refresh/${dataType}`), {
          method: "POST",
        });
        const json = await res.json().catch(() => ({}));
        if (res.ok && json.task_id) {
          await new Promise((r) => setTimeout(r, 2000));
          fetchStatus();
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Refresh failed");
      } finally {
        setRefreshing((prev) => ({ ...prev, [dataType]: false }));
      }
    },
    [fetchStatus]
  );

  if (loading && items.length === 0) {
    return (
      <div className="data-status-panel p-4 bg-gray-900 border border-gray-700 rounded-lg animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-3/4 mb-3" />
        <div className="h-4 bg-gray-700 rounded w-1/2 mb-3" />
        <div className="h-4 bg-gray-700 rounded w-5/6" />
      </div>
    );
  }

  if (error && items.length === 0) {
    return (
      <div className="data-status-panel p-4 border border-red-500/50 rounded-lg bg-gray-900">
        <p className="text-red-400">Data status unavailable. {error}</p>
        <button
          type="button"
          className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-white text-sm"
          onClick={fetchStatus}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="data-status-panel p-4 bg-gray-900 border border-gray-700 rounded-lg">
      <h2 className="text-lg font-semibold text-white mb-3">Data ingestion status</h2>
      <p className="text-gray-400 text-sm mb-4">
        Last updated per source. Celery workers run daily OHLCV, weekly macro, hourly news, quarterly fundamentals. Click Refresh to trigger now.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border border-gray-700">
          <thead>
            <tr className="bg-gray-800 text-gray-300 text-left">
              <th className="p-2 border-b border-gray-700">Source</th>
              <th className="p-2 border-b border-gray-700">Entity</th>
              <th className="p-2 border-b border-gray-700">Last updated</th>
              <th className="p-2 border-b border-gray-700">Last error</th>
              <th className="p-2 border-b border-gray-700">Action</th>
            </tr>
          </thead>
          <tbody>
            {items.length === 0 ? (
              <tr>
                <td colSpan={5} className="p-4 text-gray-500">
                  No status rows yet. Run Celery workers to populate.
                </td>
              </tr>
            ) : (
              items.map((row, i) => {
                const dt = dataTypeForRow(row);
                return (
                  <tr key={`${row.source}-${row.entity}-${i}`} className="border-b border-gray-700/50">
                    <td className="p-2 text-white">{row.source}</td>
                    <td className="p-2 text-white">{row.entity}</td>
                    <td className="p-2 text-gray-400">
                      {row.last_updated ? new Date(row.last_updated).toLocaleString() : "—"}
                    </td>
                    <td className="p-2 text-amber-400/90 max-w-xs truncate" title={row.last_error || undefined}>
                      {row.last_error || "—"}
                    </td>
                    <td className="p-2">
                      {dt ? (
                        <button
                          type="button"
                          className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded disabled:opacity-50"
                          disabled={refreshing[dt]}
                          onClick={() => handleRefresh(dt)}
                        >
                          {refreshing[dt] ? "Refreshing…" : "Refresh now"}
                        </button>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      {error && (
        <p className="mt-2 text-amber-400 text-sm">Note: {error}</p>
      )}
    </div>
  );
};
