# Particle Simulator

Hand-controlled particle simulator using webcam input.

## Requirements

- Python `3.13.5+` (developed on 3.13.5, if on any other version, remove pins from the respective requirements.txt file.)
- Webcam (`/dev/video0` on Linux/Jetson)
- For laptop backend: `mediapipe`, `opencv-python`, `numpy`
- For Jetson backend: environment/container that provides `jetson_utils` and `jetson_inference`

## Install and Run (Windows/Linux Laptop)

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

## Install and Run (Jetson Nano - Native)

1) Open terminal in project folder on Jetson.

2) Install Python dependencies:

```bash
pip3 install -r requirements.jetson.txt
```

3) In `config.py`, set:

```python
HAND_BACKEND = "jetson"
```

4) Run:

```bash
python3 main.py
```

## Install and Run (Jetson Nano - Docker)

1) Allow Docker to use display on Jetson:

```bash
xhost +local:docker
```

2) Start Jetson container (JetPack-matched image):

```bash
docker run --runtime nvidia --network host --ipc host \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/Particle_SIMULATOR:/workspace/Particle_SIMULATOR \
  -it dusty-nv/jetson-inference:r32.7.1
```

3) Inside container, install and run:

```bash
cd /workspace/Particle_SIMULATOR
pip3 install -r requirements.jetson.txt
python3 main.py
```

## Controls

- Press `q` or `Esc` to quit
- Press `r` to clear canvas
