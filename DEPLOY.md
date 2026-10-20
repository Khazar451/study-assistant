# Free public demo (Hugging Face Spaces)

Docker Spaces can be a paid feature, so the free path uses a **Gradio-SDK Space** purely as a launcher.
It runs `python app.py`, where FastAPI serves the API and the pre-built static frontend (`frontend/out`)
on port 7860. All model calls go to Nebius Token Factory, NVIDIA NIM and Tavily, so free CPU hardware is enough.

## One-time setup
1. Create a free account at huggingface.co and an access token with **write** permission (Settings, Access Tokens).
2. **New Space**: SDK **Gradio** (any template), hardware **CPU basic (free)**, visibility **Public**. Name it e.g. `study-assistant`.
3. In the Space, open **Settings, Variables and secrets** and add **secrets**:
   - `NEBIUS_API_KEY`
   - `NVIDIA_API_KEY`
   - `NVIDIA_EMBEDDER_KEY` (the embedder reads this first; slide vision reads `NVIDIA_API_KEY` first)
   - `TAVILY_API_KEY`

   and a **variable** `ENABLE_WEB_SEARCH` = `true`.

## Deploy (repeat after each commit)
Commit your work first. The script builds the frontend, publishes the last commit and leaves your branch untouched.
It needs Node.js locally (you already have it for the UI).

```bash
deploy/push_to_space.sh <hf-username>/<space-name>
```

Use your username as the Git username and the token as the password. The Space builds in a few minutes.
The demo URL is `https://<hf-username>-<space-name>.hf.space`.

## What the demo does
- `DEMO_MODE=true`: the "clear index" endpoint is disabled for visitors.
- `RATE_LIMIT_PER_MIN=20` per visitor on the endpoints that spend credits. `MAX_UPLOAD_MB=50`.
- On every start it indexes `sample_materials/astronomy_kepler.txt` (`AUTO_INGEST_SAMPLE`), so a judge can ask a question immediately.
- Storage is ephemeral: uploads and chat history reset when the Space restarts. Fine for a demo.

## Keep it awake
Free Spaces sleep after about 48 hours without visitors and wake on the next visit. Ping
`https://<hf-username>-<space-name>.hf.space/api/status` every 5 to 30 minutes with a free monitor
(UptimeRobot, cron-job.org or a scheduled GitHub Action). That endpoint calls no model, so it costs nothing.

## Docker hosts
The repo also has a `Dockerfile` (Next.js + FastAPI in one container) for any host that runs Docker
(Render, Fly.io, Nebius Serverless Endpoints). Set the same environment variables.
