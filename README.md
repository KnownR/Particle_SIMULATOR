# Particle Simulator

Hand-controlled particle simulator.

## Requirements

- Python `3.6+` (Jetson) or `3.10+` (Laptop)
- Webcam (usually `/dev/video0` on Linux/Jetson)
- For laptop: `opencv-python`, `mediapipe`, `numpy`
- For Jetson: `dustynv/jetson-inference` Docker image (provides CUDA + `jetson_utils`)

## Backends

| Backend | Where | Speed | What it does |
|---------|-------|-------|--------------|
| `auto` | (default) | — | Picks `jetson` on aarch64, `mediapipe` on x86 |
| `jetson` | Jetson Nano | 30+ FPS (CUDA) | Body pose → wrist tracking via `poseNet` |
| `mediapipe` | Laptop/Desktop | 20-30 FPS | 21-point hand landmarks (CPU) |
| `mock` | Anywhere | 60 FPS | Synthetic hand data for testing |

Set in `config.py` → `HAND_BACKEND` or via env var: `HAND_BACKEND=mock python3 main.py`

---

## Windows / Linux Laptop: Setup and Run

1) Open terminal in project folder.

2) Install dependencies:

```bash
pip install -r requirements.laptop.txt
```

3) Run:

```bash
python main.py
```

---

## Jetson Nano (Docker) — Recommended

### First Time Setup (run once, ~3 minutes)

```bash
cd /sdcard/Particle_SIMULATOR
chmod +x scripts/*.sh
./scripts/build_jetson.sh
```

This builds a Docker image (`particle-sim`) with all dependencies pre-installed.

### Run the App (instant every time)

```bash
export DISPLAY=:1
./scripts/run_jetson.sh
```

That's it. No waiting, no installs.

### Optional flags

```bash
# Custom display (default :1 for VNC)
DISPLAY=:0 ./scripts/run_jetson.sh

# Different camera
CAMERA_DEVICE=/dev/video1 ./scripts/run_jetson.sh

# Smoke test without camera
HAND_BACKEND=mock ./scripts/run_jetson.sh

# Force a specific backend
HAND_BACKEND=jetson ./scripts/run_jetson.sh
```

### How the Jetson backend works

The `jetson` backend uses `poseNet("resnet18-body")` — a body pose model that runs
on CUDA at 30+ FPS. It detects 18 body keypoints including wrist positions.

- **Wrist position** → controls particle physics (gravity, attraction, count, size)
- **Wrist height** → simulates finger count (hand high = more fingers = different mode)
- **Elbow-to-wrist angle** → controls vortex tilt

The camera should show your **upper body** (not just hands) for body pose detection.

---

## Jetson Nano (Docker) — Manual Command

If you prefer a one-liner instead of the scripts:

```bash
export DISPLAY=:1
xhost +local:docker

docker run --rm -it --runtime nvidia --network host --ipc host \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -e HAND_BACKEND=auto \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /sdcard/Particle_SIMULATOR:/workspace/Particle_SIMULATOR \
  particle-sim \
  python3 main.py
```

> **Note:** This requires `./scripts/build_jetson.sh` to have been run first.
> If you haven't built the image, replace `particle-sim` with
> `dustynv/jetson-inference:r32.7.1` — but you'll need to install deps manually.

---

## Jetson Nano (Native — no Docker)

1) Install system libraries + venv:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.jetson.txt
```

2) Run:

```bash
HAND_BACKEND=jetson python3 main.py
```

> Requires `jetson_utils` and `jetson_inference` available at system level (JetPack).

---

## Troubleshooting

- **`ImportError: libGL.so.1`**: install `libgl1` and related system libs.
- **`jetson_utils` import fails**: run `python3 scripts/diagnose_jetson.py` inside the container for full diagnostics.
- **Camera open failed**: check `ls /dev/video*` and permissions (`sudo chmod 666 /dev/video0`).
- **No window over SSH**: use VNC or `ssh -X` with a local X server.
- **Slow over VNC**: expected — lower `FPS` to `15` and resolution to `320x240` in `config.py`.
- **No body detected**: make sure upper body is visible in camera (not just hands close-up).

---

## Controls

- `q` or `Esc`: quit
- `r`: reset canvas
