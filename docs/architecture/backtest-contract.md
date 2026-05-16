# Backtest API Contract (Unified)

This document defines the current backtest contract after unification work.

## Canonical Execution Path

All public backtest endpoints now execute through:

- `core.backtest_api_adapter.run_backtest_contract`
- which delegates to `core.backtest_service.run_backtest`

This removes endpoint-specific engine behavior and keeps metrics/output formatting consistent.

## Public Endpoints

### `POST /api/v1/quant/backtest`

- Strategy-driven request (`strategy`, `fast_period`, `slow_period`, etc.)
- Uses shared strategy request contract in `api/backtest_contracts.py`

### `POST /api/v1/backtest/run`

- Legacy model-name request (`model_name`)
- `model_name` is mapped to strategy via `strategy_from_model_name`:
  - names containing `rsi` -> `rsi_mean_reversion`
  - names containing `factor`/`momentum` -> `factor_momentum`
  - otherwise -> `sma_cross`

### `POST /api/v1/backtest/technical`

- Technical SMA workflow
- Uses same shared adapter path with `sma_cross` and fast/slow parameters

## Unified Response Shape

The adapter normalizes these fields:

- `model_name`
- `symbol`
- `strategy`
- `period` (`start`, `end`)
- `metrics`
- `equity_curve`
- `trades`
- `status`

Compatibility fields are preserved so existing panels continue working.

## Regression Coverage

Tests that lock contract wiring and equivalence:

- `tests/test_backtesting_api.py`
- `tests/test_api_contract_routes.py`

These ensure route mounting and common response shape parity between `/backtest/run` and `/quant/backtest`.
