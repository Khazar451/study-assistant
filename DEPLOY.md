# Free public demo

The repo ships a `Dockerfile` that builds the UI as static files and runs everything in **one Python process**
(`app.py`: FastAPI serves the API and the UI). Memory use is about 170 to 300 MB, so it fits free 512 MB tiers.
All model calls go to Nebius Token Factory, NVIDIA NIM and Tavily, so no GPU is needed.

## Render (free web service, builds straight from GitHub)
1. Sign up at render.com with your GitHub account.
2. **New, Web Service**, pick the `Khazar451/study-assistant` repo.
3. Settings:
   - **Language:** Docker (it finds the `Dockerfile`)
   - **Instance type:** Free
   - **Health Check Path** (Advanced): `/api/status`
4. **Environment variables** (Advanced, Add environment variable):
   - `NEBIUS_API_KEY`
   - `NVIDIA_API_KEY`
   - `NVIDIA_EMBEDDER_KEY` (the embedder reads this first; slide vision reads `NVIDIA_API_KEY` first)
   - `TAVILY_API_KEY`
   - `ENABLE_WEB_SEARCH` = `true`
5. **Create Web Service.** The first build takes about 5 to 10 minutes. Your URL is `https://<service-name>.onrender.com`.
6. Every `git push` to `main` redeploys automatically.

Render fills in `PORT` itself; `app.py` reads it.

### Free-tier behavior
- Free web services spin down after a period without traffic and wake on the next request (about a minute).
  Keep it awake by pinging `https://<service-name>.onrender.com/api/status` every 10 minutes with a free monitor
  (UptimeRobot, cron-job.org or a scheduled GitHub Action). That endpoint calls no model, so it costs nothing.
- Check Render's pricing page for the current free-tier hour limits before relying on 24/7 uptime until Dec 15.
- Storage is ephemeral: uploads and chat history reset on restart or redeploy. The sample re-indexes itself on every start.

## What the demo container sets
- `DEMO_MODE=true`: the "clear index" endpoint is disabled for visitors.
- `RATE_LIMIT_PER_MIN=20` per visitor on the endpoints that spend credits. `MAX_UPLOAD_MB=50`.
- `DEMO_MAX_PER_MIN=30` and `DEMO_MAX_PER_DAY=500` across all visitors together; `ULTRA_MAX_PER_HOUR=20`.
- To stop all model spending at once, set `DEMO_PAUSED=true` in the service's environment.
- `AUTO_INGEST_SAMPLE=true`: indexes `sample_materials/astronomy_kepler.txt` so a judge can ask a question immediately.

## Other hosts
- **Any Docker host** (Fly.io, Railway, Koyeb, Nebius Serverless Endpoints): same image, same environment variables.
- **Hugging Face Spaces:** `deploy/push_to_space.sh` publishes a Gradio-SDK Space that runs `app.py`. Free CPU hardware
  was not selectable on some accounts, so this is a fallback only.
