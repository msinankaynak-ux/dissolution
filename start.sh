#!/usr/bin/env bash
# DissolvA — one command to resume work: pull latest, set up venv, run the app.
# Usage:  bash ~/dissolva/app/start.sh
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # repo root (app/)

echo "▶ Pulling latest from GitHub..."
git pull --ff-only 2>/dev/null || echo "  (skipped pull — offline or local changes)"

if [ ! -d .venv ]; then
  echo "▶ Creating virtual environment (first run)..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "▶ Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "▶ Launching DissolvA at http://localhost:8501 ..."
exec streamlit run app.py
