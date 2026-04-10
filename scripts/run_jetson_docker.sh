#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
IMAGE="${IMAGE:-dustynv/jetson-inference:r32.7.1}"
FRAMES="${FRAMES:-30}"
RUN_APP="${RUN_APP:-1}"
BACKEND="${BACKEND:-mock}"
CHECK_JETSON_INIT="${CHECK_JETSON_INIT:-0}"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"

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

if [[ "$BACKEND" != "mock" && "$BACKEND" != "jetson" && "$BACKEND" != "mediapipe" ]]; then
  fail "BACKEND must be one of: mock, jetson, mediapipe"
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

DEVICE_ARGS=()
if [[ -e "$CAMERA_DEVICE" ]]; then
  DEVICE_ARGS=(--device "$CAMERA_DEVICE")
else
  log "Camera device not found at $CAMERA_DEVICE (continuing without --device)"
fi

log "Starting container"
docker run --rm -it --runtime nvidia --network host --ipc host \
  "${DEVICE_ARGS[@]}" \
  -e HAND_BACKEND="$BACKEND" \
  -e CHECK_JETSON_INIT="$CHECK_JETSON_INIT" \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}" \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PROJECT_DIR":/workspace/Particle_SIMULATOR \
  "$IMAGE" \
  bash -lc "
set -euo pipefail
cd /workspace/Particle_SIMULATOR
python3 -m pip uninstall -y \
  opencv-python \
  opencv-contrib-python \
  opencv-python-headless \
  opencv-contrib-python-headless || true
rm -rf /usr/local/lib/python3.6/dist-packages/cv2* || true
rm -rf /usr/local/lib/python3.6/dist-packages/opencv_python* || true
rm -rf /usr/local/lib/python3.8/dist-packages/cv2* || true
rm -rf /usr/local/lib/python3.8/dist-packages/opencv_python* || true
rm -rf /usr/local/lib/python3.10/dist-packages/cv2* || true
rm -rf /usr/local/lib/python3.10/dist-packages/opencv_python* || true
apt-get update
apt-get install -y \
  python3-opencv \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:\${LD_LIBRARY_PATH:-}
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.jetson.txt
python3 -c \"import cv2; print('[PASS] cv2=', cv2.__version__)\"
python3 smoke_test.py --backend mock --frames '$FRAMES'
if [ \"\${CHECK_JETSON_INIT}\" = \"1\" ]; then
  python3 -c \"from hand_tracker import HandTracker; t=HandTracker(backend='jetson'); print('[PASS] backend=', t.backend_name)\"
else
  echo '[docker] CHECK_JETSON_INIT=0 so skipping strict jetson backend init check'
fi
if [ '$RUN_APP' = '1' ]; then
  python3 main.py
else
  echo '[docker] RUN_APP=0 so skipping main.py'
fi
"
