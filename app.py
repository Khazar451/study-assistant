"""Single-process entrypoint: FastAPI serves the API and the static frontend build.

Used by Hugging Face Gradio-SDK Spaces (which run `python app.py` and expect port 7860) and anywhere
a Docker image is not available. Locally, prefer `bin/study-assistant`.
"""

import os

import uvicorn

os.environ.setdefault("AUTO_INGEST_SAMPLE", "true")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("RATE_LIMIT_PER_MIN", "20")
os.environ.setdefault("DEMO_MAX_PER_MIN", "30")
os.environ.setdefault("DEMO_MAX_PER_DAY", "500")
os.environ.setdefault("ULTRA_MAX_PER_HOUR", "20")
os.environ.setdefault("MAX_UPLOAD_MB", "50")

from src.api.server import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
