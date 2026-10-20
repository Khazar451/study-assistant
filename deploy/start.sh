#!/usr/bin/env bash
# Start FastAPI (internal) and Next.js (public), then index the bundled sample so the demo has content.
set -e

python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 &

for _ in $(seq 1 60); do
  curl -fs http://127.0.0.1:8000/api/status >/dev/null 2>&1 && break
  sleep 1
done

# Best effort: container storage is ephemeral, so re-index the sample after every restart.
(curl -fs -X POST http://127.0.0.1:8000/api/ingest/sample >/dev/null 2>&1 || true) &

cd frontend
exec node node_modules/next/dist/bin/next start -H 0.0.0.0 -p "${PORT:-7860}"
