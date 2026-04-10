#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  build_jetson.sh  —  One-time Docker image build.
#  Run this ONCE on the Jetson. After that, ./run_jetson.sh is instant.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-particle-sim}"
BASE_IMAGE="${BASE_IMAGE:-dustynv/jetson-inference:r32.7.1}"

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
