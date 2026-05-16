# Database migrations (Alembic)

Run from repo root:

- Create migration: `alembic -c db/alembic.ini revision --autogenerate -m "message"`
- Upgrade: `alembic -c db/alembic.ini upgrade head`
- Downgrade: `alembic -c db/alembic.ini downgrade -1`
- Smoke check: `python -m db.migration_smoke_check`

Set `DATABASE_URL` in .env (e.g. `postgresql://trader:trading123@localhost:5432/trading_metrics`).

Notes:

- Startup now validates DB/Redis config before running migrations (`core/startup_validation.py`).
- Use the smoke check in CI or pre-deploy validation to ensure required tables exist after migration.
