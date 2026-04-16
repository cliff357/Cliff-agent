#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  Shannon Web — startup script
#  Usage:  ./shannon-web/start.sh [--port 8080]
# ─────────────────────────────────────────────
set -euo pipefail

PORT=${1:-8080}
# Allow --port 9000 style too
if [[ "${1:-}" == "--port" ]]; then PORT="${2:-8080}"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/venv/bin/activate"

# ── Check venv ────────────────────────────────
if [[ ! -f "$VENV" ]]; then
  echo "❌  venv not found at $VENV"
  echo "    Run:  python -m venv venv && pip install -r requirements.txt"
  exit 1
fi

# ── Check if port is already in use ──────────
EXISTING_PID=$(lsof -ti :"$PORT" 2>/dev/null || true)

if [[ -n "$EXISTING_PID" ]]; then
  EXISTING_CMD=$(ps -p "$EXISTING_PID" -o comm= 2>/dev/null || echo "unknown")
  echo "⚠️   Port $PORT 已被 PID $EXISTING_PID 佔用 ($EXISTING_CMD)"
  echo ""
  read -rp "    殺左佢重新啟動？[Y/n] " choice
  choice="${choice:-Y}"
  if [[ "$choice" =~ ^[Yy]$ ]]; then
    kill -9 "$EXISTING_PID" 2>/dev/null && echo "✅  已殺 PID $EXISTING_PID"
    sleep 1
  else
    echo "🚫  取消啟動"
    exit 0
  fi
fi

# ── Start server ──────────────────────────────
echo ""
echo "🚀  Starting Shannon Web on http://127.0.0.1:$PORT"
echo "    (Ctrl+C 停止)"
echo ""

source "$VENV"
cd "$SCRIPT_DIR"
exec python server.py --port "$PORT"
