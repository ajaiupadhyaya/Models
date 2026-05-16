"""
Celery application for data ingestion jobs.
Run from repo root so core, config, api are importable.
Broker and result backend: Redis (REDIS_URL or localhost:6379).
"""
import os
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from celery import Celery

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "financial_platform",
    broker=redis_url,
    backend=redis_url,
    include=["workers.tasks.ingestion"],
)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1,
)
# Beat schedule (cron-like)
app.conf.beat_schedule = {
    "daily-ohlcv": {
        "task": "workers.tasks.ingestion.refresh_ohlcv_daily",
        "schedule": 60 * 60 * 24,  # every 24 hours (seconds)
    },
    "weekly-macro": {
        "task": "workers.tasks.ingestion.refresh_macro_weekly",
        "schedule": 60 * 60 * 24 * 7,
    },
    "hourly-news": {
        "task": "workers.tasks.ingestion.refresh_news_hourly",
        "schedule": 60 * 60,
    },
    "quarterly-fundamentals": {
        "task": "workers.tasks.ingestion.refresh_fundamentals_quarterly",
        "schedule": 60 * 60 * 24 * 90,  # ~quarterly
    },
}
