# Terminal Improvement Plan

A phased plan to modernize the financial terminal: **UI/UX overhaul**, **reliability**, **security**, and **feature polish**. The UI refresh is a first-class track so the app feels modern and sleek end-to-end.

---

## Overview

| Phase | Focus | Duration (est.) |
|-------|--------|------------------|
| **1** | Design system + UI refresh (modern, sleek) | 1–2 weeks |
| **2** | Reliability & security | ~1 week |
| **3** | UX polish & performance | ~1 week |
| **4** | Scale & advanced features | Ongoing |

---

## Phase 1: Design System & Modern UI Refresh

**Goal:** Establish a consistent, modern, sleek look across login, shell, panels, and data so the terminal feels professional and current.

### 1.1 Design tokens and typography

- **File:** `frontend/src/styles.css`
- **Actions:**
  - Introduce a small **design token** set in `:root`:
    - Spacing scale (e.g. `--space-1` through `--space-6`)
    - Border radius scale (e.g. `--radius-sm`, `--radius-md`, `--radius-lg`)
    - Shadow scale for depth (`--shadow-panel`, `--shadow-elevated`)
    - Optional: secondary accent (e.g. blue for links/info) alongside existing orange
  - **Typography:** Keep dark theme; refine fonts:
    - Consider a slightly more distinctive sans (e.g. **DM Sans**, **Plus Jakarta Sans**, or **Geist**) for headings and UI; keep **JetBrains Mono** (or **Geist Mono**) for data and command bar.
  - Load fonts via `index.html` (Google Fonts or self-hosted) so they apply globally.

### 1.2 Login page

- **File:** `frontend/src/LoginPage.tsx` (+ CSS in `styles.css` or scoped)
- **Actions:**
  - Full-height layout with a **split or gradient background** (dark, subtle pattern or gradient) so it doesn’t feel flat.
  - **Centered card** with:
    - Softer radius (`--radius-lg`), light shadow (`--shadow-elevated`), clear hierarchy (title → subtitle → form).
  - **Inputs:** Consistent height, focus ring using accent, optional subtle placeholder styling.
  - **Primary button:** Full-width or prominent; loading state (spinner/disabled) during login.
  - **Error state:** Inline message with soft red background or border, not just text.
  - Remove or restyle “Demo” if you keep it so it doesn’t compete with primary CTA.

### 1.3 Shell and chrome

- **Files:** `frontend/src/terminal/TerminalShell.tsx`, `CommandBar.tsx`, `TickerStrip.tsx`, `styles.css`
- **Actions:**
  - **Header (terminal-header):**
    - Refine accent bar: consistent height, optional subtle bottom border or shadow; consider a **logo or wordmark** plus “Financial Terminal” (or product name).
    - Status text (e.g. “Live” / “Connected”) with a small **status dot** (green pulse when connected).
  - **Command bar:**
    - Same design tokens (radius, border, focus); optional **shortcut hint** (e.g. “⌘K”) and clear placeholder (“Type a symbol or command…”).
  - **Ticker strip:**
    - Slightly tighter typography; optional **fade at edges** (mask or gradient) for a “ticker” feel; ensure up/down colors (green/red) are consistent with design tokens.

### 1.4 Panels and content

- **Files:** `frontend/src/terminal/panels/*.tsx`, `styles.css` (`.panel`, `.panel-title`, etc.)
- **Actions:**
  - **Panel container:**
    - Use `--radius-md` (or `--radius-sm`), `--shadow-panel`, and consistent padding from spacing scale.
    - Optional very subtle **inner border** or top accent line for hierarchy.
  - **Panel titles:**
    - Slightly larger weight/size; keep uppercase + letter-spacing for “terminal” feel but align with new type scale.
  - **Tables and lists:**
    - Header row: subtle background, divider; rows: hover state; numeric columns right-aligned, `tabular-nums`.
  - **Charts (D3):**
    - Axis and grid: softer colors (e.g. `--text-soft`); tooltip: same radius/shadow as panels; ensure green/red for series match `--accent-green` / `--accent-red`.

### 1.5 Loading and error states

- **Files:** `frontend/src/terminal/panels/PanelErrorState.tsx`, each panel (e.g. `MarketOverview.tsx`), `styles.css`
- **Actions:**
  - **Loading:** Skeleton blocks (e.g. placeholder bars with light pulse animation) instead of plain “Loading…” where it makes sense (watchlist, fundamentals, charts).
  - **Errors:** Use `PanelErrorState` consistently; message + “Retry” with secondary button style; optional icon (e.g. alert) for clarity.

### 1.6 Motion and responsiveness

- **Files:** `styles.css`, panel components, `TerminalShell.tsx`
- **Actions:**
  - **Transitions:** Panel focus/active, command bar focus, button hover (e.g. 150–200ms ease).
  - **Resize handles:** If using `react-resizable-panels`, style the drag handle so it’s visible on hover and matches the new look.
  - **Responsive:** Define breakpoints (e.g. 768px, 1024px); on narrow screens consider:
    - Single-column or stacked panels, collapsible sidebar, or a tabbed layout for modules so the UI doesn’t feel cramped.

### 1.7 Deliverables (Phase 1)

- [x] Design tokens in `styles.css` (spacing, radius, shadows, optional secondary accent).
- [x] Typography and fonts wired in (Plus Jakarta Sans + JetBrains Mono); applied to login and terminal.
- [x] Login page redesigned (layout, gradient background, card with shadow, inputs, button with loading spinner, errors in styled box).
- [x] Shell chrome updated (header with gradient and status dot, command bar with ⌘K hint, ticker strip with edge fade).
- [x] All panels use shared panel styles (tokens, shadow, hover); watchlist rows and error inline use tokens.
- [x] Loading skeletons (MarketOverview) and consistent error UI (panel-error-inline, panel-error-box, PanelErrorState).
- [x] Light motion (transitions on focus/hover) and responsive behavior (breakpoints at 1024px, 768px; hint hidden on small).

---

## Phase 2: Reliability & Security

**Goal:** Harden the app for production: health checks, rate limiting, secrets, and deploy hygiene.

### 2.1 Health check and Render

- **Backend:** `api/main.py` (ensure `/health` exists and is used by Docker `HEALTHCHECK`).
- **Actions:**
  - Confirm `/health` returns 200 quickly (e.g. no heavy DB calls).
  - In Render dashboard: set **Health Check Path** to `/health` so Render marks the service unhealthy when the app is down.

### 2.2 Rate limiting

- **Backend:** `api/main.py` (middleware or dependency).
- **Actions:**
  - Add rate limiting by IP (and optionally by user after login) for `/api/*`.
  - Use a simple in-memory store or Redis (you have Redis in the project); e.g. 60–100 req/min per IP for anonymous, higher for authenticated.
  - Return 429 with a clear message when exceeded; optional `Retry-After` header.

### 2.3 Secrets and auth

- **Docs:** `DEPLOYMENT_STEP_BY_STEP.md`, `RENDER_CLI.md`.
- **Actions:**
  - Ensure all secrets (TERMINAL_USER, TERMINAL_PASSWORD, AUTH_SECRET, API keys) are only in Render Environment (never in code).
  - If AUTH_SECRET or password was ever in repo history, rotate them and document rotation in the deployment guide.
  - Optional: add a short “Security” section in the deployment doc (HTTPS, env-only secrets, optional rate limit summary).

### 2.4 Notify on fail and logs

- **Render dashboard.**
- **Actions:**
  - Enable **Notify on Fail** (email or Slack) for the service.
  - Document in `RENDER_CLI.md`: use `render logs -r <SERVICE_ID>` to debug failed deploys or runtime errors.

### 2.5 Deliverables (Phase 2)

- [x] Health check path documented in Render; `/health` lightweight and stable (no 500 on missing managers).
- [x] Rate limiting on API (IP, 100/min for `/api/*`); 429 with Retry-After; `/health`, `/docs`, etc. skipped.
- [x] Secrets verified env-only; rotation and security section in DEPLOYMENT_STEP_BY_STEP (B2.9).
- [x] Notify on Fail and CLI log usage documented (RENDER_CLI.md §8, DEPLOYMENT_STEP_BY_STEP B2.9).

---

## Phase 3: UX Polish & Performance

**Goal:** Smoother experience: caching, error copy, and basic monitoring.

### 3.1 Caching (API)

- **Backend:** Selected routers (e.g. company analysis, data, fundamentals); consider a small cache layer in `core/` or next to routers.
- **Actions:**
  - Cache expensive or rate-limited responses (e.g. FRED, Alpha Vantage, company fundamentals) in memory or Redis.
  - Short TTL (e.g. 5–15 minutes); cache key by endpoint + query (e.g. symbol, series id).
  - Document which endpoints are cached and TTL in code or a short “Caching” note in the API docs.

### 3.2 Error and rate-limit messaging (frontend)

- **Files:** Hooks (e.g. `useFetchWithRetry.ts`), panel components.
- **Actions:**
  - When API returns 429 or 5xx, show user-friendly copy (e.g. “Too many requests; try again in a minute” or “Data temporarily unavailable”).
  - Use `PanelErrorState` and optional retry-after countdown if you have that info from headers.

### 3.3 Loading and perceived performance

- **Frontend:** Panels that do heavy fetches (e.g. fundamentals, reports).
- **Actions:**
  - Ensure loading states (from Phase 1 skeletons) are visible as soon as a symbol or module is selected.
  - Optional: prefetch or cache last-used symbol data in the client to make repeat views feel instant.

### 3.4 Optional: simple metrics

- **Backend:** `api/monitoring.py` (if present) or a minimal metrics route.
- **Actions:**
  - Expose a simple metric (e.g. request count or latency bucket) for key routes so you can plug in Render or an external monitor later. No need for full Prometheus in day one.

### 3.5 Deliverables (Phase 3)

- [x] Caching for selected API endpoints (macro, market-summary, company/analyze); TTL and keys documented in `api/cache.py`.
- [x] Frontend handles 429/5xx with clear copy and retry (`useFetchWithRetry`: "Too many requests…", "Data temporarily unavailable…"; `retryAfterSeconds` for optional countdown).
- [x] Loading states (skeletons in FundamentalPanel, EconomicPanel) and optional client-side cache (30s TTL per url in `useFetchWithRetry`).
- [x] Minimal metrics endpoint: `/api/v1/monitoring/system` (request count, latency, errors).

---

## Phase 4: Scale & Advanced Features (Ongoing)

**Goal:** Prepare for more users and richer product without rewriting. Implement when needed.

### 4.1 Database (optional)

- **When:** You need persistent user data (watchlists, saved layouts, portfolios).
- **Actions:**
  - Add Render PostgreSQL (or similar); connect from API; store user preferences and optional auth extensions (e.g. multiple users, roles) later.

### 4.2 Background jobs (optional)

- **When:** Heavy work (report generation, backtests, bulk refresh) should not block HTTP.
- **Actions:**
  - Use Celery (or Render background workers) with Redis; move long-running tasks off the request path; return job id and poll or WebSocket for status.

### 4.3 Additional data sources

- **Context:** `context.md` lists Polygon, IEX, Financial Modeling Prep, etc.
- **Actions:**
  - Add one extra source (e.g. fundamentals or news); expose via existing API patterns and one new panel or section so the terminal feels more “institutional.”

### 4.4 Monitoring and alerts

- **When:** You care about uptime and debugging production.
- **Actions:**
  - Rely on Render logs + Notify on Fail; optionally add error tracking (e.g. Sentry) and a simple dashboard for health and latency.

**Phase 4 summary:** All items above are optional and can be done in any order when you need persistent user data (DB), offloaded heavy work (background jobs), more data sources, or richer monitoring. The plan is complete through Phase 3; Phase 4 is a roadmap for scaling.

---

## Implementation order (recommended)

1. **Phase 1.1–1.2:** Design tokens + typography; then login page. Delivers immediate “modern” first impression.
2. **Phase 1.3–1.5:** Shell, panels, loading/error UI. Entire app looks cohesive.
3. **Phase 1.6:** Motion and responsive. Polish.
4. **Phase 2:** Health check, rate limiting, secrets, notify. Quick wins for reliability and security.
5. **Phase 3:** Caching, error copy, loading. Better UX and API behavior.
6. **Phase 4:** As needed (DB, jobs, new data source, monitoring).

---

## File reference (quick index)

| Area | Files |
|------|--------|
| Design system | `frontend/src/styles.css`, `frontend/index.html` |
| Login | `frontend/src/LoginPage.tsx` |
| Shell / chrome | `frontend/src/terminal/TerminalShell.tsx`, `CommandBar.tsx`, `TickerStrip.tsx` |
| Panels | `frontend/src/terminal/panels/*.tsx`, `PanelErrorState.tsx` |
| API entry | `api/main.py` |
| Health | `api/main.py` (e.g. `/health`) |
| Cache | `api/cache.py` (TTLs and cached endpoints) |
| Frontend fetch | `frontend/src/hooks/useFetchWithRetry.ts` (429/5xx copy, client cache) |
| Deploy / CLI | `DEPLOYMENT_STEP_BY_STEP.md`, `RENDER_CLI.md` |

---

## Success criteria

- **UI:** Login and terminal feel modern and sleek; consistent typography, spacing, and motion; clear loading and error states.
- **Reliability:** Health check in use; rate limiting prevents abuse; secrets env-only; failures trigger notification.
- **UX:** Cached endpoints where appropriate; friendly messages on 429/5xx; fast perceived load.
- **Scale:** Clear path to DB, background jobs, and extra data sources without a rewrite.

This plan keeps the UI upgrade central while folding in reliability, security, and feature improvements in a logical order.
