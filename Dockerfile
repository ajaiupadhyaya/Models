# Stage 1: build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
# Frontend served same-origin when VITE_API_ORIGIN unset (single-service deploy)
RUN npm run build

# Stage 2: API + serve built SPA
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-api.txt ./

# Install slim API deps (required for serving). Optional heavy ML deps best-effort.
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-api.txt
ARG INSTALL_OPTIONAL_DEPS=false
RUN if [ "$INSTALL_OPTIONAL_DEPS" = "true" ]; then \
      pip install -r requirements.txt || \
      echo "WARNING: Some optional dependencies could not be installed. ML features may be limited."; \
    else \
      echo "Skipping optional heavy dependencies. Set INSTALL_OPTIONAL_DEPS=true to include notebook/ML extras."; \
    fi

COPY . .
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD sh -c 'curl -f http://127.0.0.1:${PORT:-8000}/health' || exit 1

# Single-service deploy: serves API + built SPA from same origin.
# DB/Redis are optional at boot — app degrades gracefully when unset.
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
