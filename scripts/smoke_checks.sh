#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Particle_SIMULATOR}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
FRAMES="${FRAMES:-30}"
CHECK_JETSON_INIT="${CHECK_JETSON_INIT:-1}"
CHECK_MEDIAPIPE="${CHECK_MEDIAPIPE:-0}"

log() {
  echo "[smoke] $*"
}

fail() {
  echo "[smoke][ERROR] $*" >&2
  exit 1
}

if [[ ! -d "$PROJECT_DIR" ]]; then
  fail "Project directory not found: $PROJECT_DIR"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  fail "Virtual environment missing at $VENV_DIR. Run ./scripts/setup_jetson.sh first."
fi

cd "$PROJECT_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

log "Running mock smoke test ($FRAMES frames)"
python smoke_test.py --backend mock --frames "$FRAMES"

if [[ "$CHECK_JETSON_INIT" == "1" ]]; then
  log "Checking Jetson backend initialization"
  python -c "from hand_tracker import HandTracker; t=HandTracker(backend='jetson'); print('[PASS] backend=', t.backend_name)"
fi

if [[ "$CHECK_MEDIAPIPE" == "1" ]]; then
  log "Checking MediaPipe backend initialization (optional)"
  python -c "from hand_tracker import HandTracker; t=HandTracker(backend='mediapipe'); print('[PASS] backend=', t.backend_name); t.release()"
fi

log "All selected smoke checks passed."
