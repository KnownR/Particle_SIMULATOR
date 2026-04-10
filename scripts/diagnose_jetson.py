#!/usr/bin/env python3
"""
Diagnostic script — run inside the dustynv container to debug import issues.

Usage (from Jetson host):
  docker run --rm -it --runtime nvidia \
    -v /sdcard/Particle_SIMULATOR:/workspace/Particle_SIMULATOR \
    particle-sim \
    python3 /workspace/Particle_SIMULATOR/scripts/diagnose_jetson.py
"""
import glob
import os
import sys

print("=" * 60)
print("JETSON ENVIRONMENT DIAGNOSTICS")
print("=" * 60)

print(f"\nPython:  {sys.version}")
print(f"Arch:    {os.uname().machine}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '** NOT SET **')}")

# ── Check for jetson modules on disk ──
print("\n--- Searching for jetson modules on disk ---")
search_patterns = [
    "/usr/lib/python3/dist-packages/jetson*",
    "/usr/lib/python3.6/dist-packages/jetson*",
    "/usr/local/lib/python3*/dist-packages/jetson*",
    "/jetson-inference/build/*/lib/python/jetson*",
    "/jetson-inference/build/*/lib/python/Jetson*",
]
found_any = False
for pattern in search_patterns:
    matches = glob.glob(pattern)
    for m in matches:
        print(f"  FOUND: {m}")
        found_any = True
if not found_any:
    print("  ** No jetson modules found on disk! **")

# ── Check sys.path ──
print("\n--- Python sys.path ---")
for p in sys.path:
    marker = " <-- has jetson" if os.path.exists(os.path.join(p, "jetson_utils")) or \
                                   glob.glob(os.path.join(p, "jetson_utils*")) or \
                                   os.path.exists(os.path.join(p, "jetson")) else ""
    print(f"  {p}{marker}")

# ── Try imports ──
print("\n--- Import attempts ---")
attempts = [
    ("jetson_utils",     "from jetson_utils import cudaFromNumpy"),
    ("jetson.utils",     "from jetson.utils import cudaFromNumpy"),
    ("jetson_inference",  "from jetson_inference import poseNet"),
    ("jetson.inference",  "from jetson.inference import poseNet"),
]

# Add extra paths before trying
for extra in ["/jetson-inference/build/aarch64/lib/python",
              "/usr/lib/python3/dist-packages"]:
    if os.path.isdir(extra) and extra not in sys.path:
        sys.path.insert(0, extra)
        print(f"  (added {extra} to sys.path)")

for name, code in attempts:
    try:
        exec(code)
        print(f"  [PASS] {code}")
    except Exception as e:
        print(f"  [FAIL] {code}")
        print(f"         Error: {e}")

# ── Try loading poseNet ──
print("\n--- poseNet model test ---")
try:
    try:
        from jetson_inference import poseNet
    except ImportError:
        from jetson.inference import poseNet
    net = poseNet("resnet18-body", threshold=0.15)
    print(f"  [PASS] poseNet('resnet18-body') loaded, {net.GetNumKeypoints()} keypoints")
except Exception as e:
    print(f"  [FAIL] poseNet load: {e}")

# ── OpenCV ──
print("\n--- OpenCV ---")
try:
    import cv2
    print(f"  [PASS] cv2 {cv2.__version__}")
    # Check if highgui works
    print(f"  Build info backend: {cv2.getBuildInformation().split('GUI')[0][-50:]}")
except Exception as e:
    print(f"  [FAIL] cv2: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
