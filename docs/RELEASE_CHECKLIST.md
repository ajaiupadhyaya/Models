# v1.0 Release Checklist

Use this checklist before tagging `v1.0.0` and publishing GitHub release notes.

## Automated gates

- [ ] `python -m pytest tests/test_api_contract_routes.py tests/test_db_schema_contract.py -v --tb=short`
- [ ] `python -m pytest tests/ -v --tb=short -x`
- [ ] `ruff check config/ api/ core/ --output-format=concise`
- [ ] `cd frontend && npm run test`
- [ ] `cd frontend && npm run build`

## Demo workflows (browser)

Run each command from the terminal command bar and confirm a useful result (not a blank panel or crash):

- [ ] `GP AAPL` — candlestick chart loads (live or labelled fallback)
- [ ] `QUANT AAPL` — quant panel shows factor/model content
- [ ] `PORT` — portfolio risk metrics render
- [ ] `BACKTEST AAPL` — backtest completes with equity curve and metrics
- [ ] `AI why is AAPL moving?` — AI panel responds or shows clear “not configured” message
- [ ] `HELP` — command overlay opens
- [ ] Reload page — workspace restores symbol and module

## Degraded-mode checks

- [ ] API down — terminal shows retry screen, not a blank page
- [ ] Missing `OPENAI_API_KEY` — AI panel explains limitation; other panels still work
- [ ] Missing FRED key — economic panel shows clear message, app does not crash

## Auth and deploy

- [ ] Login works when `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET` are set
- [ ] Local dev without auth (`configured: false`) opens terminal without login
- [ ] `/health` returns 200 on deployed backend
- [ ] Frontend `VITE_API_ORIGIN` points at deployed API

## Publication assets

- [ ] Screenshot: Primary instrument (`GP AAPL`)
- [ ] Screenshot: Backtest results
- [ ] Screenshot: Quant or Portfolio panel
- [ ] Optional: GIF of command-bar demo flow (Try chips → chart → backtest)
- [ ] GitHub release notes drafted (link to v2 roadmap in `docs/FEATURE_BACKLOG.md`)

## Tag and release

```bash
git tag -a v1.0.0 -m "v1.0.0 — Quant Terminal demo release"
git push origin v1.0.0
```

Then create the GitHub release with screenshots and the demo command table from `README.md`.
