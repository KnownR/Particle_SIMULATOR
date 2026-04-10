#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  build_jetson.sh  —  One-time Docker image build.
#  Run this ONCE on the Jetson. After that, ./run_jetson.sh is instant.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-particle-sim}"

# ── Detect Host L4T Version ──
# The dustynv image MUST match the host JetPack version. Otherwise, the NVIDIA
# container runtime fails to mount the host's GPU libraries over the base image stubs,
# resulting in 'file too short' import errors for libnvinfer/libcudnn.
BASE_TAG="r32.7.1" # default
if [ -f "/etc/nv_tegra_release" ]; then
    REL=$(head -n 1 /etc/nv_tegra_release | awk -F',' '{print $1}' | awk '{print $2}' | tr -d 'R')
    REV=$(head -n 1 /etc/nv_tegra_release | awk -F',' '{print $2}' | awk '{print $2}')
    if [ ! -z "$REL" ] && [ ! -z "$REV" ]; then
        BASE_TAG="r${REL}.${REV}"
    fi
fi
BASE_IMAGE="${BASE_IMAGE:-dustynv/jetson-inference:${BASE_TAG}}"
log() { echo "[build] $*"; }

log "Building image '$IMAGE_NAME' from base '$BASE_IMAGE'..."
log "This takes 2-3 minutes the first time (cached after that)."

docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$IMAGE_NAME" \
  -f "$PROJECT_DIR/Dockerfile.jetson" \
  "$PROJECT_DIR"

log "✓ Image '$IMAGE_NAME' is ready."
log "Run the app with: ./scripts/run_jetson.sh"
