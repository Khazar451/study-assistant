# Single-process deployment (Render, Fly, Nebius Serverless Endpoints, any Docker host).
# Stage 1 builds the Next.js UI as static files; stage 2 is Python only: FastAPI serves the API and
# those files on $PORT (app.py), so no Node process runs in production (keeps RAM under 512 MB).

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN STATIC_EXPORT=1 npm run build

FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src
COPY evaluation ./evaluation
COPY sample_materials ./sample_materials
COPY --from=web /web/out ./frontend/out
RUN mkdir -p chroma_db data && chown -R user:user /app

USER user
ENV PORT=7860 \
    DEMO_MODE=true \
    RATE_LIMIT_PER_MIN=20 \
    DEMO_MAX_PER_MIN=30 \
    DEMO_MAX_PER_DAY=500 \
    ULTRA_MAX_PER_HOUR=20 \
    MAX_UPLOAD_MB=50 \
    AUTO_INGEST_SAMPLE=true \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
