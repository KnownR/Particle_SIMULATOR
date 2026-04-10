#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
BACKEND="${BACKEND:-jetson}"

log() {
  echo "[setup] $*"
}

fail() {
  echo "[setup][ERROR] $*" >&2
  exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "This script is intended for Linux/Jetson."
fi

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found. Install Python 3 first."
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  fail "Project directory not found: $PROJECT_DIR"
fi

log "Project dir: $PROJECT_DIR"
cd "$PROJECT_DIR"

log "Installing required system libraries (sudo may prompt)."
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  python3-opencv \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1

log "Creating virtual environment: $VENV_DIR"
python3 -m venv --system-site-packages "$VENV_DIR"

log "Activating virtual environment and installing Python dependencies"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.jetson.txt

if [[ "$BACKEND" != "jetson" && "$BACKEND" != "mock" && "$BACKEND" != "mediapipe" ]]; then
  fail "BACKEND must be one of: jetson, mock, mediapipe"
fi

log "Setup complete."
log "Next step: run ./scripts/smoke_checks.sh"
log "If needed, set backend in config.py to: HAND_BACKEND = \"$BACKEND\""
