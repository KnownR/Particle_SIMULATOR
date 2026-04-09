#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
IMAGE="${IMAGE:-dusty-nv/jetson-inference:r32.7.1}"
FRAMES="${FRAMES:-30}"
RUN_APP="${RUN_APP:-1}"

log() {
  echo "[docker] $*"
}

fail() {
  echo "[docker][ERROR] $*" >&2
  exit 1
}

if [[ ! -d "$PROJECT_DIR" ]]; then
  fail "Project directory not found: $PROJECT_DIR"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found. Install Docker on Jetson first."
fi

if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:0
  log "DISPLAY not set. Using DISPLAY=:0"
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

log "Pulling image: $IMAGE"
docker pull "$IMAGE"

log "Starting container"
docker run --rm -it --runtime nvidia --network host --ipc host \
  --device /dev/video0 \
  -e HAND_BACKEND=jetson \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PROJECT_DIR":/workspace/Particle_SIMULATOR \
  "$IMAGE" \
  bash -lc "
set -euo pipefail
cd /workspace/Particle_SIMULATOR
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.jetson.txt
python3 smoke_test.py --backend mock --frames '$FRAMES'
python3 -c \"from hand_tracker import HandTracker; t=HandTracker(backend='jetson'); print('[PASS] backend=', t.backend_name)\"
if [ '$RUN_APP' = '1' ]; then
  python3 main.py
else
  echo '[docker] RUN_APP=0 so skipping main.py'
fi
"
