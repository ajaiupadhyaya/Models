#!/usr/bin/env bash
# One-command launch: build and start the terminal with Docker, then open the browser.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Starting terminal (Docker)..."
docker-compose up --build -d
echo "Waiting for app to be ready..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "App is up. Opening http://localhost:8000"
    if command -v open >/dev/null 2>&1; then
      open http://localhost:8000
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open http://localhost:8000
    else
      echo "Open http://localhost:8000 in your browser and sign in."
    fi
    exit 0
  fi
  sleep 2
done
echo "App may still be starting. Open http://localhost:8000 in your browser and sign in."
exit 0
