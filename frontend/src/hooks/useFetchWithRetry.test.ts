import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useFetchWithRetry } from "./useFetchWithRetry";

describe("useFetchWithRetry", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("returns loading then data on 200", async () => {
    const mockData = { id: 1 };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockData),
    });

    const { result } = renderHook(() => useFetchWithRetry<{ id: number }>("/api/test"));

    expect(result.current.loading).toBe(true);
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBeNull();
  });

  it("sets error on 4xx and does not retry", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ detail: "Not found" }),
    });

    const { result } = renderHook(() => useFetchWithRetry("/api/test", { maxRetries: 2 }));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.error).toContain("Not found");
    expect(result.current.data).toBeNull();
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  it("parse returning null sets error", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ detail: "bad" }),
    });

    const { result } = renderHook(() =>
      useFetchWithRetry("/api/test", {
        parse: (json) => ((json as { detail?: string }).detail ? null : (json as object)),
      })
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.error).toBeTruthy();
    expect(result.current.data).toBeNull();
  });

  it("retry() triggers refetch", async () => {
    let callCount = 0;
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation(() => {
      callCount++;
      return Promise.resolve({
        ok: callCount === 1 ? false : true,
        status: callCount === 1 ? 500 : 200,
        json: () => Promise.resolve(callCount === 1 ? {} : { data: "ok" }),
      });
    });

    const { result } = renderHook(() => useFetchWithRetry<{ data: string }>("/api/test", { maxRetries: 0 }));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.error).toBeTruthy();
    expect(callCount).toBe(1);

    result.current.retry();
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    }, { timeout: 3000 });
    expect(callCount).toBeGreaterThanOrEqual(2);
  });
});
