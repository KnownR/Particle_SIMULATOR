# Particle Simulator

Hand-controlled particle simulator.

## Requirements

- Python `3.10+`
- Webcam (usually `/dev/video0` on Linux/Jetson)
- For laptop run: `opencv-python`, `mediapipe`, `numpy`
- For Jetson backend: `jetson_utils` and `jetson_inference` (from Jetson environment/container)

## Backends

- `HAND_BACKEND = "mediapipe"` for laptop
- `HAND_BACKEND = "jetson"` for Jetson Nano
- `HAND_BACKEND = "mock"` for smoke testing without camera/model

## One-Command Jetson Scripts (Recommended)

From project root on Jetson:

```bash
chmod +x scripts/*.sh
```

1) Full setup (system libs + venv + deps):

```bash
./scripts/setup_jetson.sh
```

2) Smoke checks (mock + jetson backend init):

```bash
./scripts/smoke_checks.sh
```

3) Docker run (pull image, run smoke checks in container, then app):

```bash
./scripts/run_jetson_docker.sh
```

Optional flags:

```bash
FRAMES=60 ./scripts/smoke_checks.sh
RUN_APP=0 ./scripts/run_jetson_docker.sh
IMAGE=dustynv/jetson-inference:r32.7.1 ./scripts/run_jetson_docker.sh
```

---

## Windows / Linux Laptop: Setup and Run

1) Open terminal in project folder.

2) Install dependencies:

```bash
pip install -r requirements.laptop.txt
```

3) In `config.py`, set:

```python
HAND_BACKEND = "mediapipe"
```

4) Run:

```bash
python main.py
```

---

## Jetson Nano (Native): Setup and Run

1) Open terminal on Jetson and go to project:

```bash
cd ~/Particle_SIMULATOR
```

2) Install system libraries + Python 3.8 (the dustynv container ships Python 3.6;
   mediapipe aarch64 wheels require Python >= 3.8):

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.8 python3.8-dev python3.8-distutils
```

3) Bootstrap pip for Python 3.8 and create a venv:

```bash
python3.8 -m ensurepip --upgrade 2>/dev/null || (wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && python3.8 /tmp/get-pip.py)
python3.8 -m pip install virtualenv
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

4) Install Python dependencies:

```bash
pip install -r requirements.jetson.txt
```

5) `config.py` already defaults to:

```python
HAND_BACKEND = "mediapipe"
```

6) Smoke test first (no camera needed):

```bash
python smoke_test.py --backend mock --frames 30
```

Expected:

```text
[PASS] Smoke test passed for backend='mock' over 30 frames.
```

7) Check MediaPipe backend init:

```bash
python -c "from hand_tracker import HandTracker; t=HandTracker(); print(t.backend_name)"
```

8) Run full app:

```bash
python main.py
```

---

## Jetson Nano (Docker): Setup and Run

### A) If Jetson has monitor attached (recommended)

1) SSH into Jetson:

```bash
ssh <user>@<jetson-ip>
```

2) Set display + allow docker X access:

```bash
export DISPLAY=:0
xhost +local:docker
```

3) Start container (use image tag matching your JetPack):

```bash
docker run --rm -it --runtime nvidia --network host --ipc host \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/Particle_SIMULATOR:/workspace/Particle_SIMULATOR \
  dustynv/jetson-inference:r32.7.1
```

4) Inside container:

```bash
cd /workspace/Particle_SIMULATOR
python3 -m pip install -r requirements.jetson.txt
python3 smoke_test.py --backend mock --frames 30
python3 -c "from hand_tracker import HandTracker; t=HandTracker(backend='jetson'); print(t.backend_name)"
python3 main.py
```

### B) SSH-only (no monitor)

`cv2.imshow` window will not appear in plain SSH shell.
Use one of these:

- X11 forwarding (`ssh -X`) + local X server
- VNC/NoMachine remote desktop session on Jetson

---

## Troubleshooting

- `ImportError: libGL.so.1`: install `libgl1` and related system libs.
- `Cannot import mediapipe`: you are on Python 3.6 — mediapipe needs Python 3.8+. Install via deadsnakes PPA (see native setup step 2).
- Camera open failed: check `ls /dev/video0` and camera permissions (`sudo chmod 666 /dev/video0`).
- No window shown over SSH: use VNC or `ssh -X` with a local X server.
- Slow over VNC: expected — VNC adds latency. Lower `FPS` in `config.py` to `15` to reduce load.

---

## Controls

- `q` or `Esc`: quit
- `r`: reset canvas
