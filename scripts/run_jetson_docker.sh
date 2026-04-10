#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  run_jetson_docker.sh  —  LEGACY one-shot Docker run.
#
#  ⚠ DEPRECATED: Use ./build_jetson.sh + ./run_jetson.sh instead.
#  This script pulls the raw dustynv image and installs deps every run.
#  The new workflow builds a custom image once, then runs instantly.
#
#  Kept for reference / fallback only.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
IMAGE="${IMAGE:-dustynv/jetson-inference:r32.7.1}"
FRAMES="${FRAMES:-30}"
RUN_APP="${RUN_APP:-1}"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
export DISPLAY="${DISPLAY:-:1}"

log() {
  echo "[docker] $*"
}

fail() {
  echo "[docker][ERROR] $*" >&2
  exit 1
}

echo "============================================================"
echo "  ⚠  This script installs deps on every run (slow)."
echo "  ⚠  Use ./build_jetson.sh + ./run_jetson.sh instead."
echo "============================================================"

if [[ ! -d "$PROJECT_DIR" ]]; then
  fail "Project directory not found: $PROJECT_DIR"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found. Install Docker on Jetson first."
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

log "Pulling image: $IMAGE"
docker pull "$IMAGE"

DEVICE_ARGS=()
if [[ -e "$CAMERA_DEVICE" ]]; then
  DEVICE_ARGS=(--device "$CAMERA_DEVICE")
else
  log "Camera device not found at $CAMERA_DEVICE (continuing without --device)"
fi

log "Starting container"
docker run --rm -it --runtime nvidia --network host --ipc host \
  "${DEVICE_ARGS[@]}" \
  -e HAND_BACKEND=auto \
  -e DISPLAY="$DISPLAY" \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PROJECT_DIR":/workspace/Particle_SIMULATOR \
  "$IMAGE" \
  bash -lc "
set -euo pipefail
cd /workspace/Particle_SIMULATOR

# ── Remove pip-installed OpenCV (conflicts with APT system package) ──
python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null || true
rm -rf /usr/local/lib/python3.6/dist-packages/cv2* /usr/local/lib/python3.6/dist-packages/opencv_python* 2>/dev/null || true

# ── System runtime libs ──
apt-get update -qq
apt-get install -y python3-opencv libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:\${LD_LIBRARY_PATH:-}
python3 -m pip install --upgrade pip
python3 -m pip install 'numpy>=1.19,<2.0'

# ── Sanity checks ──
python3 -c \\\"import cv2; print('[PASS] cv2=', cv2.__version__)\\\"
python3 smoke_test.py --backend mock --frames $FRAMES

if [ '$RUN_APP' = '1' ]; then
  python3 main.py
else
  echo '[docker] RUN_APP=0; skipping main.py'
fi
"
