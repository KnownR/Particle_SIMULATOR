# ─────────────────────────────────────────
#  hand_tracker.py  —  Cross-platform tracker facade
#  Laptop: MediaPipe backend (CPU)
#  Jetson: jetson-inference poseNet body backend (CUDA)
#  Mock:   synthetic data for smoke testing
# ─────────────────────────────────────────

import math
import os
import platform
import sys
import urllib.request

from config import (
    HAND_BACKEND,
    JETSON_POSE_MODEL,
    JETSON_POSE_THRESHOLD,
    PINCH_THRESHOLD,
)

try:
    import cv2
    _CV2_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _ensure_mediapipe_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand_landmarker.task (~6 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV import failed. Install system OpenGL libs first (e.g. libgl1, "
            "libglib2.0-0) and ensure cv2 is installed in this environment."
        ) from _CV2_IMPORT_ERROR


def _draw_landmarks(frame, landmarks_px):
    if cv2 is None:
        return
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 200, 100), 1)


def _count_fingers(pts):
    fingers = []
    fingers.append(pts[4][0] < pts[3][0])
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(pts[tip][1] < pts[pip][1])
    return fingers


def _extract_from_points(pts, w, h):
    if len(pts) < 21:
        return None

    palm_x = sum(pts[i][0] for i in [0, 1, 5, 9, 13, 17]) / 6
    palm_y = sum(pts[i][1] for i in [0, 1, 5, 9, 13, 17]) / 6

    fingers_up = _count_fingers(pts)
    finger_count = sum(fingers_up)

    pinch_dist = math.dist(pts[4], pts[8])
    is_pinched = pinch_dist < PINCH_THRESHOLD

    wrist = pts[0]
    mid_mcp = pts[9]
    angle_rad = math.atan2(mid_mcp[1] - wrist[1], mid_mcp[0] - wrist[0])
    tilt_deg = math.degrees(angle_rad)

    fingertips_norm = [pts[i] for i in [4, 8, 12, 16, 20]]
    fingertips_px = [(int(x * w), int(y * h)) for x, y in fingertips_norm]

    return {
        "palm_norm": (palm_x, palm_y),
        "palm_px": (int(palm_x * w), int(palm_y * h)),
        "finger_count": int(finger_count),
        "fingers_up": fingers_up,
        "is_pinched": bool(is_pinched),
        "pinch_dist": float(pinch_dist),
        "tilt_deg": float(tilt_deg),
        "fingertips_px": fingertips_px,
        "index_tip_px": fingertips_px[1],
    }


# ─────────────────────────────────────────
#  Jetson import helper — tries every known path/name
# ─────────────────────────────────────────

def _try_import_jetson():
    """Try to import jetson_utils and jetson_inference.

    The dustynv containers install them to /usr/lib/python3/dist-packages/.
    Older containers use underscore names (jetson_utils), newer ones use
    the jetson.utils namespace.  We try both, and also add known build
    paths to sys.path before retrying.

    Returns (jetson_utils_module, jetson_inference_module) or raises ImportError.
    """
    # Add known search paths for jetson Python bindings
    extra_paths = [
        "/jetson-inference/build/aarch64/lib/python",
        "/usr/lib/python3/dist-packages",
    ]
    for p in extra_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Ensure LD_LIBRARY_PATH includes where jetson-inference .so files live.
    # Without this, 'import jetson_utils' finds the .pyd/.so but it fails
    # to load because libjetson-inference.so (in /usr/local/lib) isn't found.
    ld_extra = [
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu/tegra",
    ]
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    for p in ld_extra:
        if os.path.isdir(p) and p not in current_ld:
            current_ld = p + ":" + current_ld
    os.environ["LD_LIBRARY_PATH"] = current_ld

    errors = []

    # Attempt 1: underscore form (older containers, r32.x)
    try:
        import jetson_utils as _ju
        import jetson_inference as _ji
        return _ju, _ji
    except ImportError as e:
        errors.append(f"jetson_utils: {e}")

    # Attempt 2: namespace package form (newer containers)
    try:
        import jetson.utils as _ju      # type: ignore
        import jetson.inference as _ji   # type: ignore
        return _ju, _ji
    except ImportError as e:
        errors.append(f"jetson.utils: {e}")

    raise ImportError(
        f"Could not import jetson bindings.\n"
        f"  Tried: {'; '.join(errors)}\n"
        f"  LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}\n"
        f"  Relevant sys.path entries: "
        f"{[p for p in sys.path if 'jetson' in p.lower() or 'dist-packages' in p]}\n"
        f"  Fix: run inside dustynv/jetson-inference container with --runtime nvidia.\n"
        f"  Debug: python3 scripts/diagnose_jetson.py"
    )


# ─────────────────────────────────────────
#  MediaPipe backend  (CPU — laptop & fallback)
# ─────────────────────────────────────────

class _MediaPipeBackend:
    def __init__(self):
        _require_cv2()
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe backend selected, but mediapipe is not installed."
            ) from exc

        _ensure_mediapipe_model()
        self.mp = mp
        self.mp_vision = mp_vision
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._ts_ms = 0
        self.name = "mediapipe"

    def process(self, frame):
        self._ts_ms += 33

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        left_data = None
        right_data = None

        for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
            raw_label = handedness_list[0].category_name
            label = "Right" if raw_label == "Left" else "Left"

            pts = [(float(lm.x), float(lm.y)) for lm in lm_list]
            landmarks_px = [(int(x * w), int(y * h)) for x, y in pts]
            _draw_landmarks(frame, landmarks_px)

            data = _extract_from_points(pts, w, h)
            if data is None:
                continue

            if label == "Left":
                left_data = data
            else:
                right_data = data

        return {"left": left_data, "right": right_data}

    def release(self):
        self.landmarker.close()


# ─────────────────────────────────────────
#  Jetson backend  (CUDA — GPU-accelerated body pose)
# ─────────────────────────────────────────
#
#  Uses poseNet("resnet18-body") which detects 18 body keypoints via CUDA.
#  We extract wrist positions (left_wrist=9, right_wrist=10) as hand locations.
#
#  Finger count and gestures are approximated from wrist height:
#    hand high → many fingers up = high count
#    hand low  → fist/few fingers = low count
#
#  This is NOT 21-point hand landmark detection — it's body-pose wrist
#  tracking.  But it runs at 30+ FPS on the Jetson Nano's GPU, which
#  beats 8 FPS CPU-only MediaPipe over VNC.
#
#  Body keypoints (resnet18-body):
#   0  nose           5  left_shoulder   9  left_wrist    13  left_knee
#   1  left_eye       6  right_shoulder  10 right_wrist   14  right_knee
#   2  right_eye      7  left_elbow      11 left_hip      15  left_ankle
#   3  left_ear       8  right_elbow     12 right_hip     16  right_ankle
#   4  right_ear                                          17  neck
# ─────────────────────────────────────────

class _JetsonBackend:

    # Map wrist-Y (normalised) → simulated finger count.
    # Hand raised high = more fingers up.  Hand low = fist.
    _FINGER_ZONES = [
        (0.20, 5),   # very high  → 5 fingers (rainbow palette / wave mode)
        (0.35, 4),   # high       → 4 (aurora / wave)
        (0.50, 3),   # mid-upper  → 3 (ocean / chaos)
        (0.65, 2),   # mid-lower  → 2 (fire / orbital)
        (0.80, 1),   # low        → 1 (mono / wind)
        (1.00, 0),   # very low   → 0 (freeze)
    ]

    def __init__(self):
        _require_cv2()

        ju, ji = _try_import_jetson()
        self._cudaFromNumpy = ju.cudaFromNumpy
        self._cudaDeviceSynchronize = ju.cudaDeviceSynchronize

        model_name = os.environ.get("JETSON_POSE_MODEL", JETSON_POSE_MODEL)
        threshold = float(os.environ.get("JETSON_POSE_THRESHOLD", JETSON_POSE_THRESHOLD))

        self.net = ji.poseNet(model_name, threshold=threshold)
        self.name = f"jetson:{model_name}"
        print(f"[INFO] JetsonBackend loaded: {self.name} "
              f"({self.net.GetNumKeypoints()} keypoints, CUDA)")

    # ── per-frame ──────────────────────────
    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cuda_img = self._cudaFromNumpy(rgb)

        poses = self.net.Process(cuda_img)
        self._cudaDeviceSynchronize()

        left_data = None
        right_data = None

        for pose in poses:
            self._draw_pose(frame, pose)

            ld = self._hand_from_wrist(pose, w, h, "left")
            rd = self._hand_from_wrist(pose, w, h, "right")

            if ld and left_data is None:
                left_data = ld
            if rd and right_data is None:
                right_data = rd

        return {"left": left_data, "right": right_data}

    # ── extract hand-like data from a body wrist keypoint ──
    def _hand_from_wrist(self, pose, w, h, side):
        wrist_idx = pose.FindKeypoint(f"{side}_wrist")
        if wrist_idx < 0:
            return None

        kp = pose.Keypoints[wrist_idx]
        # poseNet returns pixel coordinates
        nx = kp.x / max(w, 1)
        ny = kp.y / max(h, 1)

        # Finger count from wrist height
        finger_count = 0
        for max_y, fc in self._FINGER_ZONES:
            if ny < max_y:
                finger_count = fc
                break

        fingers_up = [i < finger_count for i in range(5)]

        # Tilt from elbow → wrist angle
        tilt_deg = -90.0
        elbow_idx = pose.FindKeypoint(f"{side}_elbow")
        if elbow_idx >= 0:
            ek = pose.Keypoints[elbow_idx]
            tilt_deg = math.degrees(math.atan2(kp.y - ek.y, kp.x - ek.x))

        # Pinch: not possible with body pose — always False
        palm_px = (int(kp.x), int(kp.y))
        tip_px = (int(kp.x), max(int(kp.y) - 20, 0))

        return {
            "palm_norm": (nx, ny),
            "palm_px": palm_px,
            "finger_count": int(finger_count),
            "fingers_up": fingers_up,
            "is_pinched": False,
            "pinch_dist": 0.5,
            "tilt_deg": float(tilt_deg),
            "fingertips_px": [tip_px] * 5,
            "index_tip_px": tip_px,
        }

    # ── draw upper-body keypoints for visual feedback ──
    def _draw_pose(self, frame, pose):
        if cv2 is None:
            return
        arm_joints = ["shoulder", "elbow", "wrist"]
        for side in ("left", "right"):
            color = (0, 255, 0) if side == "left" else (255, 160, 0)
            pts = []
            for joint in arm_joints:
                idx = pose.FindKeypoint(f"{side}_{joint}")
                if idx >= 0:
                    kp = pose.Keypoints[idx]
                    px = (int(kp.x), int(kp.y))
                    cv2.circle(frame, px, 5, color, -1)
                    pts.append(px)
            # Connect joints with lines
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], (0, 200, 100), 2)

    def release(self):
        pass


# ─────────────────────────────────────────
#  Mock backend  (no camera needed)
# ─────────────────────────────────────────

class _MockBackend:
    def __init__(self):
        self.name = "mock"
        self._tick = 0

    def process(self, frame):
        h, w = frame.shape[:2]
        self._tick += 1

        t = self._tick * 0.08
        cx = 0.5 + 0.2 * math.sin(t)
        cy = 0.5 + 0.15 * math.cos(t * 0.7)

        left = self._make_hand(
            center_x=min(max(cx - 0.18, 0.1), 0.9),
            center_y=min(max(cy, 0.1), 0.9),
            w=w,
            h=h,
            finger_count=3,
        )
        right = self._make_hand(
            center_x=min(max(cx + 0.18, 0.1), 0.9),
            center_y=min(max(cy, 0.1), 0.9),
            w=w,
            h=h,
            finger_count=2,
        )

        return {"left": left, "right": right}

    def _make_hand(self, center_x, center_y, w, h, finger_count):
        palm_px = (int(center_x * w), int(center_y * h))
        index_tip_px = (int((center_x + 0.04) * w), int((center_y - 0.07) * h))

        fingers_up = [False] * 5
        for i in range(min(max(finger_count, 0), 5)):
            fingers_up[i] = True

        return {
            "palm_norm": (float(center_x), float(center_y)),
            "palm_px": palm_px,
            "finger_count": int(finger_count),
            "fingers_up": fingers_up,
            "is_pinched": False,
            "pinch_dist": 0.2,
            "tilt_deg": -90.0,
            "fingertips_px": [index_tip_px] * 5,
            "index_tip_px": index_tip_px,
        }

    def release(self):
        pass


# ─────────────────────────────────────────
#  Backend resolution
# ─────────────────────────────────────────

def _resolve_backend_name(backend_name):
    if backend_name is None:
        backend_name = os.environ.get("HAND_BACKEND", HAND_BACKEND)

    backend_name = backend_name.lower().strip()
    if backend_name == "auto":
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64"):
            return "jetson"
        return "mediapipe"
    return backend_name


class HandTracker:
    def __init__(self, backend=None):
        resolved = _resolve_backend_name(backend)

        if resolved == "mediapipe":
            self._impl = _MediaPipeBackend()

        elif resolved == "jetson":
            try:
                self._impl = _JetsonBackend()
            except (RuntimeError, ImportError) as exc:
                print("=" * 60)
                print("  !! JETSON BACKEND FAILED !!")
                print(f"  Error: {exc}")
                print("=" * 60)
                try:
                    print("[WARN] Trying mediapipe fallback...")
                    self._impl = _MediaPipeBackend()
                except (RuntimeError, ImportError):
                    print("=" * 60)
                    print("  !! MEDIAPIPE ALSO FAILED !!")
                    print("  !! USING MOCK BACKEND -- NO REAL TRACKING !!")
                    print("  !! You will see fake hands moving on their own !!")
                    print("=" * 60)
                    self._impl = _MockBackend()

        elif resolved == "mock":
            self._impl = _MockBackend()

        else:
            raise ValueError(
                f"Unknown HAND_BACKEND='{resolved}'. Expected auto|mediapipe|jetson|mock"
            )

        self.backend_name = self._impl.name
        print(f"[INFO] HandTracker backend: {self.backend_name}")

    def process(self, frame):
        return self._impl.process(frame)

    def release(self):
        self._impl.release()