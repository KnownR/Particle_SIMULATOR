#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  run_jetson.sh  —  Launch the Particle Simulator instantly.
#  Requires: ./build_jetson.sh was run at least once.
#
#  Usage:
#    ./scripts/run_jetson.sh                     # default: auto backend, DISPLAY=:1
#    DISPLAY=:0 ./scripts/run_jetson.sh          # custom display
#    CAMERA_DEVICE=/dev/video1 ./scripts/run_jetson.sh
#    HAND_BACKEND=mock ./scripts/run_jetson.sh   # smoke test without camera
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
IMAGE_NAME="${IMAGE_NAME:-particle-sim}"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
HAND_BACKEND="${HAND_BACKEND:-auto}"
export DISPLAY="${DISPLAY:-:1}"

log() { echo "[run] $*"; }

# ── Auto-build if image doesn't exist ──
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  log "Image '$IMAGE_NAME' not found. Building first..."
  "$SCRIPT_DIR/build_jetson.sh"
fi

# ── X11 access for the container ──
if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

# ── Camera device ──
DEVICE_ARGS=()
if [[ -e "$CAMERA_DEVICE" ]]; then
  DEVICE_ARGS=(--device "$CAMERA_DEVICE")
  log "Camera: $CAMERA_DEVICE"
else
  log "⚠ Camera $CAMERA_DEVICE not found (continuing without it)"
fi

log "Display: $DISPLAY | Backend: $HAND_BACKEND"
log "Starting Particle Simulator..."

exec docker run --rm -it --runtime nvidia --network host --ipc host \
  ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
  -e DISPLAY="$DISPLAY" \
  -e HAND_BACKEND="$HAND_BACKEND" \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PROJECT_DIR":/workspace/Particle_SIMULATOR \
  "$IMAGE_NAME" \
  python3 main.py
