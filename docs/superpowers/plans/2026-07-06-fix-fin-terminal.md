# FIN-TERMINAL Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return the Models quant terminal (github.com/ajaiupadhyaya/Models) to a fully usable state by completing tonight's half-finished Stitch UI migration and fixing the backend/CI defects found during diagnosis.

**Architecture:** FastAPI backend (`api/`, entrypoint `backend/main.py` → `api.main:app`) + Vite/React 18/TypeScript/D3 SPA (`frontend/`), served same-origin from a Fly.io machine (`models-terminal`, 512MB). We fix FORWARD (keep the Stitch design), not revert: restore the deleted CSS custom-property token layer with Stitch palette values so D3 charts and non-migrated panels get colors back, fix the flex `min-width` layout bug, then replace the Tailwind Play CDN with a real Tailwind build.

**Tech Stack:** Python 3.12 (use `.venv-ci/bin/python`; repo's `.venv` is broken — no `bin/`), pytest, FastAPI/Pydantic v2; Node 22, Vite 8, React 18, Tailwind CSS v3 (to be added), react-resizable-panels v4, D3 v7; Fly.io (`flyctl` authed), GitHub Actions CI.

## Diagnosis summary (verified 2026-07-06, all claims tested live)

The app is functionally healthy — data plumbing, command bar (`GP TSLA` works), live news, 372/372 backend tests, 24/24 frontend tests, frontend builds — but three commits tonight (`398f0864` → `0b400ba3` → `8cc58969`) half-migrated the UI to Stitch's Tailwind-Play-CDN design system and broke presentation:

1. **Charts render black-on-black.** The rewrite gutted `frontend/src/styles.css` `:root` to just `color-scheme: dark`, deleting every `--*` token. ~40 D3 call sites (e.g. `charts/candlestickHelpers.ts:50` `fill = "var(--accent-green)"`) and ~880 surviving lines of legacy CSS still consume those vars; unresolved `var()` in SVG fill = black.
2. **Columns overlap/clip.** The old `<Panel style={{minWidth: 0}}>` was replaced with Tailwind classes that omit `min-w-0` (`frontend/src/terminal/TerminalShell.tsx:343,349,355`); flex items can't shrink below intrinsic width, so the center column (80px `<h1>` at `PrimaryInstrument.tsx:651`, full-width SVGs) overflows under its neighbors. Also `main` uses `mt-16` (64px) but `TopNavBar` is ~72px tall.
3. **Tailwind Play CDN in index.html** (`frontend/index.html:7`) — dev-only tool, console-warns, unusable for production. Inline config at `index.html:12-95` is a byte-for-byte copy of the Stitch export config (tokens ARE all defined; the custom classes resolve fine).
4. **`GET /api/v1/reports/health` → 500** (`api/investor_reports_api.py:409`): return annotation `Dict[str, bool]` but handler returns `"status": "healthy"` (str) → Pydantic `bool_parsing` → 500. Frontend polls it in a loop. Bug exists in prod too.
5. **`GET /api/v1/company/analyze/{ticker}` → 500**: `Out of range float values are not JSON compliant: nan` — unsanitized NaN in the response dict (`api/company_analysis_api.py`).
6. **CI red on main**: `tests/test_automation.py` fails in CI with `ModuleNotFoundError: No module named 'schedule'` — `schedule>=1.2.0` is in `requirements.txt`/`requirements-api.txt` but missing from `requirements-ci.txt` (which CI installs). Passes locally only because `.venv-ci` happens to have it.
7. Minor: `SideNavBar.tsx:8-11` labels don't match the modules they open (CRYPTO→`technical`, FIXED→`quant`, RESEARCH→`fundamental`); env-name drift `ALPACA_API_BASE` (`.env`, 3 API files) vs `ALPACA_BASE_URL` (`core/data_fetcher_enhanced.py:590`, `.env.example`); favicon 404; `npx tsc --noEmit` = 92 pre-existing errors (74 from missing `@types/d3`) that Vite ignores.

**Deploy state:** Fly app `models-terminal` is up/healthy serving the PRE-overhaul build. Vercel: no deployment (frontend ships inside the Fly image via 2-stage Dockerfile; keep it that way). Design references: `stitch_screens/screen{1-4}_*.{html,png}` (untracked — do not delete).

**Fallback (do not execute unless fix-forward fails):** last-good commit is `51eb297a`; `git revert 8cc58969 0b400ba3 398f0864` fully undoes the restyle.

## Global Constraints

- Python: always `.venv-ci/bin/python` with `PYTHONPATH=.` from repo root (or recreate `.venv` via `uv venv --python 3.12 .venv && uv pip install -r requirements-api.txt -r requirements-ci.txt` — user standard is uv).
- Do NOT revert or rewrite the three UI commits; fix forward on `main`.
- Do NOT delete `stitch_screens/` (untracked design references) or archive-worthy code — archive, don't delete.
- Do NOT enable `SCHEDULER_ENABLED` or set `DATABASE_URL`; DB-less degradation is by design.
- Never print `.env` values; it contains real keys.
- Stitch palette anchors: bg `#131313`, panel `#1c1b1b`, hairline `#4c4546`, text `#e5e2e1`, muted `#988e90`, accent blue `#3568ff`, radius `0px`, fonts Inter + Space Mono.
- Every task ends with the full relevant test suite green before commit.
- A local backend may already be running on :8000 and Vite dev servers on :5173/:5174 from the diagnosis session — reuse or kill-and-restart YOUR OWN servers freely, but don't assume ports are free.

---

## Phase 1 — Backend correctness (independent of UI work)

### Task 1: Fix `/api/v1/reports/health` 500

**Files:**
- Modify: `api/investor_reports_api.py:409`
- Test: `tests/test_investor_reports_health.py` (create)

**Interfaces:**
- Produces: `GET /api/v1/reports/health` → 200 `{"status": "healthy"|"error", "openai_configured": bool}`. Frontend polls this path; no frontend change needed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_investor_reports_health.py`:

```python
"""Regression test: /api/v1/reports/health must not 500 (bool_parsing bug)."""

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_reports_health_returns_200_with_string_status(client):
    response = client.get("/api/v1/reports/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("healthy", "error")
    assert isinstance(body["openai_configured"], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/test_investor_reports_health.py -q`
Expected: FAIL — response is 500 (ResponseValidationError: bool_parsing on `'healthy'`).

- [ ] **Step 3: Fix the return annotation**

In `api/investor_reports_api.py` line 409, change:

```python
async def investor_reports_health() -> Dict[str, bool]:
```

to:

```python
async def investor_reports_health() -> Dict[str, Any]:
```

Check the file's imports: `from typing import ... ` — add `Any` if not already imported.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/test_investor_reports_health.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add api/investor_reports_api.py tests/test_investor_reports_health.py
git commit -m "fix: /api/v1/reports/health 500 — status is a string, not bool"
```

### Task 2: Fix `/api/v1/company/analyze/{ticker}` NaN 500

**Files:**
- Modify: `api/company_analysis_api.py` (handler `analyze_company` at line 114; helper added above it)
- Test: `tests/test_company_api.py` (append one test; follow the existing mock pattern at lines 32-56)

**Interfaces:**
- Produces: `_json_safe(value) -> Any` module-level helper in `api/company_analysis_api.py` — recursively maps non-finite floats (NaN/±Inf, incl. numpy float subclasses) to `None`. `GET /api/v1/company/analyze/{ticker}` returns 200 with `null` where NaN was.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_company_api.py`:

```python
def test_analyze_sanitizes_nan_to_null(client):
    """NaN in analyzer output must serialize as null, not 500 (regression)."""
    with patch("api.company_analysis_api.CompanySearch") as MockSearch:
        MockSearch.return_value.validate_ticker.return_value = (True, "OK")
        with patch("api.company_analysis_api.CompanyAnalyzer") as MockAnalyzer:
            mock_analyzer = MagicMock()
            mock_analyzer.comprehensive_analysis.return_value = {
                "profile": {"name": "NaN Test Inc"},
                "ratios": {"pe_ratio": float("nan"), "pb_ratio": float("inf")},
                "financials": {},
            }
            MockAnalyzer.return_value = mock_analyzer
            response = client.get(
                # unique ticker so the 15-min analyze cache can't serve a stale entry
                "/api/v1/company/analyze/NANTST?include_dcf=false&include_risk=false&include_technicals=false"
            )
    assert response.status_code == 200
    ratios = response.json()["fundamental_analysis"]["ratios"]
    assert ratios["pe_ratio"] is None
    assert ratios["pb_ratio"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/test_company_api.py -q`
Expected: new test FAILS with 500 (`ValueError: Out of range float values are not JSON compliant`); the two existing tests still pass.

- [ ] **Step 3: Add sanitizer and apply it**

In `api/company_analysis_api.py`, add near the top (after imports; ensure `import math` is present):

```python
def _json_safe(value):
    """Recursively replace non-finite floats (NaN/Inf) with None — they are not JSON compliant."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
```

In `analyze_company`, locate where the finished `response` dict is cached/returned (the `set_cached(key, response)` call near the end of the handler) and insert immediately before it:

```python
        response = _json_safe(response)
```

so both the cache and the HTTP response hold sanitized data.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/test_company_api.py -q`
Expected: all pass (3 tests).

- [ ] **Step 5: Live-verify against the real endpoint**

Start (or reuse) the backend: `PYTHONPATH=. .venv-ci/bin/python -m uvicorn backend.main:app --port 8000` then:
Run: `curl -s http://127.0.0.1:8000/api/v1/company/analyze/AAPL | head -c 300`
Expected: JSON starting `{"ticker":"AAPL","company_name":...` — NOT `{"detail":"Out of range float values..."}`. (Needs network for yfinance; if the provider is flaky, the mocked test is the gate.)

- [ ] **Step 6: Commit**

```bash
git add api/company_analysis_api.py tests/test_company_api.py
git commit -m "fix: sanitize NaN/Inf in company analyze response (500 -> 200)"
```

### Task 3: Un-break CI (missing `schedule` in requirements-ci.txt)

**Files:**
- Modify: `requirements-ci.txt` (add one line in the "Reporting/utilities" area)

**Interfaces:**
- Produces: CI backend job (installs `requirements-ci.txt`, runs `pytest tests/ -v -x`) passes `tests/test_automation.py`.

- [ ] **Step 1: Reproduce CI's failure locally with a clean CI-parity venv**

```bash
cd ~/Documents/Models
uv venv /tmp/models-ci-venv --python 3.12
uv pip install -p /tmp/models-ci-venv -r requirements-ci.txt
PYTHONPATH=. /tmp/models-ci-venv/bin/python -m pytest tests/test_automation.py -x -q
```

Expected: FAIL `ModuleNotFoundError: No module named 'schedule'` (matches CI run 28834420667).

- [ ] **Step 2: Add the missing dependency**

In `requirements-ci.txt`, after the `# Reporting/utilities` block header, add:

```
schedule>=1.2.0
```

- [ ] **Step 3: Verify in the CI-parity venv, then run the whole suite there**

```bash
uv pip install -p /tmp/models-ci-venv -r requirements-ci.txt
PYTHONPATH=. /tmp/models-ci-venv/bin/python -m pytest tests/ -x -q
```

Expected: `372 passed, 17 skipped` (or current counts + Task 1/2 additions), zero failures. If ANOTHER module is missing, add it to `requirements-ci.txt` the same way and re-run until green — this venv is exactly what CI sees.

- [ ] **Step 4: Commit**

```bash
git add requirements-ci.txt
git commit -m "ci: add schedule to requirements-ci.txt (test_automation ModuleNotFoundError)"
```

### Task 4: Reconcile `ALPACA_API_BASE` vs `ALPACA_BASE_URL`

**Files:**
- Modify: `core/data_fetcher_enhanced.py:590` and the error string at `:737`
- Modify: `.env.example` (rename `ALPACA_BASE_URL` → `ALPACA_API_BASE`)

**Interfaces:**
- Produces: all code paths honor `ALPACA_API_BASE` (the name actually set in `.env` and used by `api/paper_trading_api.py:99`, `api/automation_api.py:45`, `core/automated_trading_orchestrator.py:117`), with `ALPACA_BASE_URL` kept as a fallback alias.

- [ ] **Step 1: Make the fetcher read the canonical name with legacy fallback**

In `core/data_fetcher_enhanced.py` line 590, change:

```python
            base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
```

to:

```python
            base_url = os.getenv("ALPACA_API_BASE") or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
```

At line 737, update the guidance string `"Set ALPACA_API_KEY, ALPACA_API_SECRET, and ALPACA_BASE_URL in .env file. "` → `"Set ALPACA_API_KEY, ALPACA_API_SECRET, and ALPACA_API_BASE in .env file. "`.

In `.env.example`, rename the `ALPACA_BASE_URL` key to `ALPACA_API_BASE` (same example value).

- [ ] **Step 2: Verify nothing regressed**

Run: `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/test_data_providers.py tests/test_unified_fetcher.py -q` then the full suite `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/ -q`.
Expected: all green. Also `grep -rn "ALPACA_BASE_URL" api core config` → only the fallback in `data_fetcher_enhanced.py` remains.

- [ ] **Step 3: Commit**

```bash
git add core/data_fetcher_enhanced.py .env.example
git commit -m "fix: standardize on ALPACA_API_BASE env var (keep legacy fallback)"
```

---

## Phase 2 — Frontend stabilization (make the site usable tonight)

All frontend tasks: work in `frontend/`; verify with `npm run build` + `npm test` + a dev-server screenshot. Keep the backend from Phase 1 running on :8000 so `/api` proxying works during visual checks.

### Task 5: Restore the CSS design-token layer (fixes black-on-black charts + dead legacy panels)

**Files:**
- Modify: `frontend/src/styles.css:1-11` (the `:root` block and `body` rule)

**Interfaces:**
- Produces: CSS custom properties `--bg, --bg-secondary, --bg-panel, --border, --border-focus, --text, --text-soft, --accent, --accent-hover, --accent-green, --accent-red, --accent-blue, --font-sans, --font-mono, --space-1..6, --radius-sm/md/lg, --shadow-panel, --shadow-elevated, --transition-fast, --transition-normal` — the exact names consumed by `frontend/src/charts/*` (~40 sites) and every non-migrated `panels/*` component — now valued from the Stitch palette. Do NOT rename any variable; only define them.

- [ ] **Step 1: Replace the gutted `:root` and `body` block**

In `frontend/src/styles.css`, replace lines 1-11 (`:root { color-scheme: dark; }` through the current `body { ... }`) with:

```css
:root {
  color-scheme: dark;
  /* Legacy token names, remapped to the Stitch palette.
     D3 charts and non-migrated panels consume these via var(--...). */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --radius-sm: 0px;
  --radius-md: 0px;
  --radius-lg: 0px;
  --shadow-panel: none;
  --shadow-elevated: none;
  --transition-fast: 100ms linear;
  --transition-normal: 150ms linear;
  --bg: #131313;
  --bg-secondary: #1c1b1b;
  --bg-panel: #0e0e0e;
  --border: #4c4546;
  --border-focus: #3568ff;
  --text: #e5e2e1;
  --text-soft: #988e90;
  --accent: #3568ff;
  --accent-hover: #5c7cff;
  --accent-green: #22c55e;
  --accent-red: #ef4444;
  --accent-blue: #3568ff;
  --font-sans: "Inter", system-ui, -apple-system, sans-serif;
  --font-mono: "Space Mono", ui-monospace, Menlo, Consolas, monospace;
}

body {
  margin: 0;
  background-color: #131313;
  color: #e5e2e1;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}
```

(Radius 0 and shadow none are deliberate — the Stitch system is flat/square. Body bg moves from `#000` to `#131313` to stop fighting Tailwind's `bg-background`.)

- [ ] **Step 2: Verify build + visual**

Run: `npm run build && npm test` in `frontend/` — expect clean build, 24 tests pass.
Start `npm run dev`, open the served URL, run `GP AAPL`. Expected: candlesticks now green/red, volume bars visible, axes/labels legible; PORTFOLIO tab's risk/stress bar charts colored (blue accent), not black blobs.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/styles.css
git commit -m "fix: restore CSS design tokens (Stitch palette) — charts and legacy panels get colors back"
```

### Task 6: Fix panel overflow/overlap (`min-w-0`) and the giant symbol heading

**Files:**
- Modify: `frontend/src/terminal/TerminalShell.tsx:343,349,350,355`
- Modify: `frontend/src/terminal/panels/PrimaryInstrument.tsx:651` and the toolbar div ~line 653

**Interfaces:**
- Consumes: react-resizable-panels v4 `Panel` (flex-basis-% items; need `min-width: 0` to shrink).
- Produces: no API change — layout-only classNames.

- [ ] **Step 1: Add `min-w-0` to all three Panels and the center wrapper**

In `frontend/src/terminal/TerminalShell.tsx`:
- line 343: `className="overflow-auto bg-surface-container-low hairline-r"` → `className="min-w-0 overflow-auto bg-surface-container-low hairline-r"`
- line 349: `className="overflow-y-auto bg-background"` → `className="min-w-0 overflow-y-auto bg-background"`
- line 350: `className="min-h-full flex flex-col"` → `className="min-h-full min-w-0 flex flex-col"`
- line 355: `className="overflow-auto bg-surface-container-low hairline-l"` → `className="min-w-0 overflow-auto bg-surface-container-low hairline-l"`

- [ ] **Step 2: Clamp the display heading and make the chart toolbar wrap**

In `frontend/src/terminal/panels/PrimaryInstrument.tsx` line 651, change:

```tsx
<h1 className="font-display-price text-[80px] leading-[80px] tracking-[-0.04em] font-extrabold text-on-surface uppercase">{primarySymbol}</h1>
```

to:

```tsx
<h1 className="font-display-price text-[clamp(36px,6vw,80px)] leading-[1.0] tracking-[-0.04em] font-extrabold text-on-surface uppercase break-all">{primarySymbol}</h1>
```

Two lines below, replace the toolbar's inline style `style={{ display: "flex", gap: 4, alignItems: "center" }}` with `className="flex flex-wrap gap-1 items-center justify-end"` (delete the `style` prop) so PNG/SVG/SMA/timeframe chips wrap instead of overflowing under the right panel.

- [ ] **Step 3: Verify**

`npm run build && npm test` green. In the dev server: center column no longer slides under the watchlist or AI panel; `VOL:`/`CLOSE:` readouts fully visible; drag the panel separators — center shrinks gracefully.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/terminal/TerminalShell.tsx frontend/src/terminal/panels/PrimaryInstrument.tsx
git commit -m "fix: min-w-0 on resizable panels + clamp display heading — stop column overlap"
```

### Task 7: Fix top/side chrome offsets (content under the header)

**Files:**
- Modify: `frontend/src/terminal/TopNavBar.tsx:8`
- Modify: `frontend/src/terminal/SideNavBar.tsx:15`
- Modify: `frontend/src/terminal/TerminalShell.tsx:328`

**Interfaces:**
- Produces: fixed header exactly 80px (`h-20`), side rail starts below it; `main` offset matches.

- [ ] **Step 1: Pin the header height**

`TopNavBar.tsx:8`: add `h-20` to the header className (`"fixed top-0 left-0 w-full z-50 h-20 bg-background ..."` — keep the rest, drop `py-4` since the fixed height + `items-center` handles vertical centering).

- [ ] **Step 2: Offset the side rail below the header**

`SideNavBar.tsx:15`: change `"fixed left-0 top-0 h-full w-20 flex flex-col items-center py-24 z-40 ..."` → `"fixed left-0 top-20 h-[calc(100vh-80px)] w-20 flex flex-col items-center py-6 z-40 ..."` (keep the remaining classes).

- [ ] **Step 3: Match the main offset**

`TerminalShell.tsx:328`: change `ml-20 mt-16 ... h-[calc(100vh-64px)]` → `ml-20 mt-20 ... h-[calc(100vh-80px)]` (leave every other class in that className untouched).

- [ ] **Step 4: Verify + commit**

`npm run build` green; dev server: ticker strip fully visible below the header at any window width; no dead band or overlap at the top of the watchlist.

```bash
git add frontend/src/terminal/TopNavBar.tsx frontend/src/terminal/SideNavBar.tsx frontend/src/terminal/TerminalShell.tsx
git commit -m "fix: align fixed chrome offsets (header 80px, side rail below header)"
```

### Task 8: Make SideNavBar labels match what they open

**Files:**
- Modify: `frontend/src/terminal/SideNavBar.tsx:7-12`

**Interfaces:**
- Consumes: `TerminalContext.setActiveModule` module ids `"primary" | "technical" | "quant" | "fundamental"` (see `MainContent` switch in `TerminalShell.tsx:114-167`).
- Produces: labels truthfully name the module. There are no crypto/fixed-income modules — don't invent them.

- [ ] **Step 1: Fix the labels**

Replace `SideNavBar.tsx` lines 7-12 with:

```tsx
  const navItems = [
    { id: "primary", icon: "trending_up", label: "EQUITIES" },
    { id: "technical", icon: "insights", label: "TECHNICAL" },
    { id: "quant", icon: "functions", label: "QUANT" },
    { id: "fundamental", icon: "description", label: "RESEARCH" },
  ] as const;
```

- [ ] **Step 2: Verify + commit**

Dev server: each rail icon opens the panel its label claims (TECHNICAL → technical panel, QUANT → quant panel).

```bash
git add frontend/src/terminal/SideNavBar.tsx
git commit -m "fix: side-nav labels match the modules they open (no phantom crypto/fixed)"
```

### Task 9: End-to-end visual verification gate (Phase 2 exit)

**Files:** none (verification only; fix regressions in place if found)

- [ ] **Step 1: Full local stack + walkthrough**

Backend on :8000 (Phase 1 build), `npm run dev` in `frontend/`. With Playwright or by hand, walk: login page → `GP AAPL` → `QUANT AAPL` → `PORT` → `BACKTEST AAPL` → NEWS tab → each side-rail icon. Screenshot each.

- [ ] **Step 2: Acceptance checklist**

- No black-on-black charts anywhere (candles green/red, risk/stress bars colored).
- No column overlap at 1200px and 1600px widths; separators draggable.
- Browser console: zero 500s (`/api/v1/reports/health` returns 200 — Task 1), zero uncaught errors. (The Tailwind-CDN production warning is still expected until Task 10.)
- Compare side-by-side with `stitch_screens/screen1_gallery.png` — same family: flat/square, hairlines, Inter display type, Space Mono labels, blue accent.
- `npm test` and `PYTHONPATH=. .venv-ci/bin/python -m pytest tests/ -q` both green.

- [ ] **Step 3: Commit any straggler fixes, then push Phases 1-2**

```bash
git push origin main
gh run watch --repo ajaiupadhyaya/Models   # CI must go green (Task 3 fixed the backend job)
```

---

## Phase 3 — Productionize (kill the Play CDN, TypeScript hygiene)

### Task 10: Real Tailwind build (replace cdn.tailwindcss.com)

**Files:**
- Create: `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/tailwind.css`
- Modify: `frontend/index.html` (remove lines 7 and 12-95; add favicon), `frontend/src/main.tsx` (import tailwind.css), `frontend/package.json` (devDeps)

**Interfaces:**
- Produces: identical utility classes compiled at build time; `index.html` free of any `cdn.tailwindcss.com` reference. The custom token names (`bg-background`, `text-on-surface-variant`, `font-label-xs`, `p-margin-page`, `gap-gutter`, …) must keep working — every migrated component depends on them.

- [ ] **Step 1: Install Tailwind v3 toolchain**

```bash
cd frontend
npm install -D tailwindcss@^3.4 postcss autoprefixer @tailwindcss/forms @tailwindcss/container-queries
```

- [ ] **Step 2: Create `frontend/tailwind.config.js`**

Copy the theme verbatim from `index.html:13-94` (it is the Stitch export config). Full file:

```js
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "surface-container-lowest": "#0e0e0e",
        "on-tertiary-container": "#3568ff",
        "surface-container": "#201f1f",
        "primary-fixed": "#e2e2e2",
        "tertiary-container": "#000000",
        "surface": "#131313",
        "secondary-fixed-dim": "#c6c6c7",
        "on-error-container": "#ffdad6",
        "on-surface": "#e5e2e1",
        "outline-variant": "#4c4546",
        "tertiary-fixed-dim": "#b6c4ff",
        "surface-container-high": "#2a2a2a",
        "on-primary-container": "#757575",
        "secondary": "#c6c6c7",
        "primary-fixed-dim": "#c6c6c6",
        "inverse-surface": "#e5e2e1",
        "on-primary-fixed-variant": "#474747",
        "surface-dim": "#131313",
        "on-tertiary-fixed-variant": "#0039b3",
        "inverse-primary": "#5e5e5e",
        "error": "#ffb4ab",
        "tertiary-fixed": "#dce1ff",
        "surface-container-low": "#1c1b1b",
        "inverse-on-surface": "#313030",
        "surface-variant": "#353534",
        "on-error": "#690005",
        "tertiary": "#b6c4ff",
        "secondary-container": "#454747",
        "outline": "#988e90",
        "on-tertiary-fixed": "#001551",
        "on-secondary-container": "#b4b5b5",
        "background": "#131313",
        "surface-tint": "#c6c6c6",
        "surface-bright": "#3a3939",
        "on-background": "#e5e2e1",
        "surface-container-highest": "#353534",
        "on-tertiary": "#002780",
        "on-primary": "#303030",
        "on-secondary": "#2f3131",
        "secondary-fixed": "#e2e2e2",
        "primary-container": "#000000",
        "on-surface-variant": "#cfc4c5",
        "error-container": "#93000a",
        "on-secondary-fixed": "#1a1c1c",
        "primary": "#c6c6c6",
        "on-secondary-fixed-variant": "#454747",
        "on-primary-fixed": "#1b1b1b"
      },
      borderRadius: { DEFAULT: "0px", lg: "0px", xl: "0px", full: "0px" },
      spacing: { "margin-page": "48px", "cell-padding": "12px", "unit": "4px", "gutter": "1px" },
      fontFamily: {
        "body-md": ["Inter", "sans-serif"],
        "display-price": ["Inter", "sans-serif"],
        "headline-lg": ["Inter", "sans-serif"],
        "data-mono": ["Space Mono", "monospace"],
        "label-xs": ["Space Mono", "monospace"]
      },
      fontSize: {
        "body-md": ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "display-price": ["120px", { lineHeight: "110px", letterSpacing: "-0.06em", fontWeight: "800" }],
        "headline-lg": ["32px", { lineHeight: "40px", letterSpacing: "-0.02em", fontWeight: "700" }],
        "data-mono": ["12px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "400" }],
        "label-xs": ["10px", { lineHeight: "12px", fontWeight: "700" }]
      }
    }
  },
  plugins: [require("@tailwindcss/forms"), require("@tailwindcss/container-queries")]
};
```

- [ ] **Step 3: Create `frontend/postcss.config.js` and `frontend/src/tailwind.css`**

`postcss.config.js`:

```js
module.exports = { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`src/tailwind.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

In `frontend/src/main.tsx`, add `import "./tailwind.css";` immediately BEFORE the existing `import "./styles.css";` (styles.css must stay last so its hairline/scrollbar/legacy rules win ties).

- [ ] **Step 4: Strip the CDN from `index.html` and add a favicon**

Delete line 7 (`<script src="https://cdn.tailwindcss.com...">`) and the whole `<script id="tailwind-config">...</script>` block (lines 12-95). Keep the font `<link>`s. Add in `<head>`:

```html
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' fill='%23131313'/><text x='2' y='12' font-family='monospace' font-size='10' fill='%233568ff'>F</text></svg>" />
```

- [ ] **Step 5: Verify parity**

```bash
npm run build && npm test
grep -r "cdn.tailwindcss" dist/ index.html && echo "CDN STILL PRESENT" || echo "CDN gone"
npm run preview   # serves dist/
```

Expected: build green (CSS asset grows to ~30-60kB — compiled utilities), tests green, "CDN gone". Open the preview URL: pixel-parity with the dev look from Task 9 (spot-check header, watchlist, candles, portfolio). Console: the Tailwind production warning is GONE.

- [ ] **Step 6: Commit**

```bash
git add frontend/tailwind.config.js frontend/postcss.config.js frontend/src/tailwind.css frontend/src/main.tsx frontend/index.html frontend/package.json frontend/package-lock.json
git commit -m "build: replace Tailwind Play CDN with real Tailwind v3 build + favicon"
```

### Task 11: TypeScript hygiene — make `tsc --noEmit` pass and gate it in CI

**Files:**
- Modify: `frontend/package.json` (add `@types/d3` devDep + `typecheck` script)
- Modify: `frontend/src/terminal/panels/QuantPanel.tsx`, `ScreeningPanel.tsx`, `AiAssistantPanel.tsx`, `frontend/src/charts/AreaChart.tsx`, `TimeSeriesLine.tsx` (the ~18 non-d3 errors)
- Modify: `.github/workflows/ci.yml` (frontend job: add typecheck step)

**Interfaces:**
- Produces: `npm run typecheck` (= `tsc --noEmit`) exits 0; CI frontend job runs it.

- [ ] **Step 1: Install d3 types — kills 74 of 92 errors**

```bash
cd frontend && npm install -D @types/d3
npx tsc --noEmit 2>&1 | wc -l   # expect ~18 remaining
```

Add to `package.json` scripts: `"typecheck": "tsc --noEmit"`.

- [ ] **Step 2: Fix the remaining errors, mechanically, one file at a time**

Current inventory (re-derive with `npx tsc --noEmit`):
- `QuantPanel.tsx` — `TS2339 'trades' does not exist on BacktestResult`: add `trades?: number;` (or the actual payload type) to the `BacktestResult` interface where it's declared in that file/types module. 7× `TS18048 's.metrics' possibly undefined`: use `s.metrics?.<field> ?? 0` at each site.
- `ScreeningPanel.tsx` — 5× `TS2367` comparисon `SortKey` vs `"sparkline"`: add `"sparkline"` to the `SortKey` union type; `TS2345` follows from the same fix.
- `AiAssistantPanel.tsx:165,167` — `TS2322 unknown → ReactNode`: wrap in `String(...)`.
- `AreaChart.tsx:129` / `TimeSeriesLine.tsx:129` — `TS2683 'this' implicitly any`: type the D3 each/on callback as `function (this: SVGElement, evt: PointerEvent) {...}` or convert to an arrow fn using the bound selection.
After each file: `npm run typecheck` — error count strictly decreasing; `npm test` still green.

- [ ] **Step 3: Gate in CI**

In `.github/workflows/ci.yml`, in the `frontend` job after the install step, add:

```yaml
      - name: Typecheck
        run: npm run typecheck
        working-directory: frontend
```

- [ ] **Step 4: Verify + commit**

`npm run typecheck` → exit 0. `npm run build && npm test` → green.

```bash
git add frontend/package.json frontend/package-lock.json frontend/src .github/workflows/ci.yml
git commit -m "chore: @types/d3 + fix remaining TS errors + typecheck gate in CI"
```

---

## Phase 4 — Ship

### Task 12: Push, CI green, deploy to Fly, live verification

**Files:** none new (deploy + verify)

- [ ] **Step 1: Full local gate**

```bash
cd ~/Documents/Models
PYTHONPATH=. .venv-ci/bin/python -m pytest tests/ -q          # expect all pass
cd frontend && npm run typecheck && npm test && npm run build  # expect all green
```

- [ ] **Step 2: Push and watch CI**

```bash
git push origin main
gh run watch --repo ajaiupadhyaya/Models
```

Expected: backend ✓ (Task 3), lint ✓, frontend ✓ (now incl. typecheck). Do not deploy on red.

- [ ] **Step 3: Deploy to Fly (Docker builds the frontend fresh — nothing extra to do)**

```bash
flyctl deploy -a models-terminal
```

- [ ] **Step 4: Live verification (cold start can take ~20s — scale-to-zero)**

```bash
curl -s https://models-terminal.fly.dev/health                       # 200 healthy
curl -s -o /dev/null -w '%{http_code}\n' https://models-terminal.fly.dev/api/v1/reports/health   # 200 (was 500)
curl -s https://models-terminal.fly.dev/ | grep -c "cdn.tailwindcss"  # 0
```

Open https://models-terminal.fly.dev/ in a browser: login → `GP AAPL` → confirm the Stitch-styled terminal renders with colored charts, no overlap, clean console.

- [ ] **Step 5: Commit any release notes and stop**

If `docs/RELEASE_CHECKLIST.md` needs a line for this release, add it; otherwise done.

---

## Explicitly deferred (follow-up plan, do NOT start here)

- Migrate the ~15 legacy panels (`Fundamental, Technical, Quant, Economic, News, NewsSentiment, Backtest, Optimizer, StressTest, PaperTrading, Automation, Screening, AiInsights, DataStatus, TickerStrip, PanelErrorState`) off `styles.css` `terminal-*`/`panel-*` classes onto Tailwind tokens, then delete the dead ~880 lines of legacy CSS. The Task-5 token bridge keeps them looking correct until then.
- Refactor D3 charts to take a theme/colors prop (from `charts/theme.ts`) instead of raw `var(--…)` strings.
- Track `stitch_screens/` in git (or move to `docs/design/`) with provenance.
- Self-host the Inter/Space Mono/Material Symbols fonts (removes the last external `<link>`s; Google Fonts responses vary by user-agent so SRI hashes can't be applied to them — self-hosting via `@fontsource/*` packages is the proper fix). Task 10 already removes the only external *script*.
- Optional Vercel frontend split (`frontend/vercel.json` exists, no deployment) — YAGNI while Fly serves the SPA same-origin.
- Repo hygiene: `render.yaml`, `docker-compose.yml`, `workers/celery_app.py` are unused alternates — archive decision for later, don't delete now.
