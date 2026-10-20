# Single-container deployment (Hugging Face Spaces Docker SDK, Render, Fly, any Docker host).
# Next.js serves the UI on $PORT (default 7860) and proxies /api/* to FastAPI on 127.0.0.1:8000.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=web /usr/local/bin/node /usr/local/bin/node
COPY --from=web /web /app/frontend
COPY src ./src
COPY evaluation ./evaluation
COPY sample_materials ./sample_materials
COPY deploy/start.sh ./start.sh
RUN chmod +x start.sh && mkdir -p chroma_db data && chown -R user:user /app

USER user
ENV PORT=7860 \
    API_URL=http://127.0.0.1:8000 \
    DEMO_MODE=true \
    RATE_LIMIT_PER_MIN=20 \
    MAX_UPLOAD_MB=50 \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["./start.sh"]
