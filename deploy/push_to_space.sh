#!/usr/bin/env bash
# Publish the last COMMIT of this repo to a free Hugging Face Space without touching your branch.
# The Space uses the Gradio SDK only as a launcher: it runs `python app.py`, where FastAPI serves
# the API plus the static Next.js build (frontend/out).
# Usage: deploy/push_to_space.sh <hf-username>/<space-name>
# When git asks for a password, paste a Hugging Face access token with write permission.
set -euo pipefail

SPACE="${1:?usage: deploy/push_to_space.sh <hf-username>/<space-name>}"
ROOT="$(git rev-parse --show-toplevel)"

echo "Building static frontend..."
(cd "$ROOT/frontend" && npm ci --silent && STATIC_EXPORT=1 npm run build >/dev/null)
[ -f "$ROOT/frontend/out/index.html" ] || { echo "Frontend build failed"; exit 1; }

TMP="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$TMP" 2>/dev/null || true' EXIT
git -C "$ROOT" worktree add -q --detach "$TMP" HEAD
cd "$TMP"

mkdir -p frontend && cp -r "$ROOT/frontend/out" frontend/out

# Spaces read their config from the README front matter; keep it out of the GitHub README.
{
  cat <<'FM'
---
title: Study Assistant
emoji: 🎓
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
python_version: "3.11"
app_file: app.py
license: mit
short_description: Grounded RAG tutor on NVIDIA Nemotron via Nebius Token Factory
---

FM
  cat README.md
} > README.space && mv README.space README.md

git add -A && git add -f frontend/out
git -c user.name="deploy" -c user.email="deploy@localhost" commit -q -m "Deploy to Hugging Face Space"
git push --force "https://huggingface.co/spaces/${SPACE}" HEAD:main
echo "Pushed. Build logs: https://huggingface.co/spaces/${SPACE}?logs=build"
