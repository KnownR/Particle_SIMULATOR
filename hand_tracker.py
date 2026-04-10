# ─────────────────────────────────────────
#  hand_tracker.py  —  Cross-platform tracker facade
#  Laptop: MediaPipe backend
#  Jetson: jetson-inference pose backend
# ─────────────────────────────────────────

import math
import os
import platform
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


class _JetsonBackend:
    def __init__(self):
        _require_cv2()
        try:
            from jetson_utils import cudaFromNumpy, cudaDeviceSynchronize
            from jetson_inference import poseNet
        except ImportError as exc:
            raise RuntimeError(
                "Jetson backend selected, but jetson-inference bindings are missing. "
                "Run inside a Jetson container/environment that provides jetson_utils "
                "and jetson_inference."
            ) from exc

        model_name = os.environ.get("JETSON_POSE_MODEL", JETSON_POSE_MODEL)
        threshold = float(os.environ.get("JETSON_POSE_THRESHOLD", JETSON_POSE_THRESHOLD))

        self.cudaFromNumpy = cudaFromNumpy
        self.cudaDeviceSynchronize = cudaDeviceSynchronize
        self.net = poseNet(model_name, threshold=threshold)
        self.name = f"jetson:{model_name}"

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cuda_img = self.cudaFromNumpy(rgb)

        poses = self.net.Process(cuda_img)
        self.cudaDeviceSynchronize()

        left_data = None
        right_data = None

        for pose in poses:
            pts = self._extract_pose_points(pose, w, h)
            if pts is None:
                continue

            landmarks_px = [(int(x * w), int(y * h)) for x, y in pts]
            _draw_landmarks(frame, landmarks_px)
            data = _extract_from_points(pts, w, h)
            if data is None:
                continue

            palm_x = data["palm_norm"][0]
            label = "Left" if palm_x < 0.5 else "Right"

            if label == "Left":
                left_data = data
            else:
                right_data = data

        return {"left": left_data, "right": right_data}

    def _extract_pose_points(self, pose, w, h):
        keypoints = getattr(pose, "Keypoints", None)
        if not keypoints:
            return None

        pts = [None] * 21
        for kp in keypoints:
            kp_id = getattr(kp, "ID", getattr(kp, "id", None))
            x = getattr(kp, "x", None)
            y = getattr(kp, "y", None)
            if kp_id is None or x is None or y is None:
                continue
            if 0 <= kp_id < 21:
                pts[kp_id] = (float(x) / max(w, 1), float(y) / max(h, 1))

        if any(p is None for p in pts):
            return None
        return pts

    def release(self):
        pass


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


def _resolve_backend_name(backend_name):
    if backend_name is None:
        backend_name = os.environ.get("HAND_BACKEND", HAND_BACKEND)

    backend_name = backend_name.lower().strip()
    if backend_name == "auto":
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64"):
            return "mediapipe"
        return "mediapipe"
    return backend_name


class HandTracker:
    def __init__(self, backend=None):
        resolved = _resolve_backend_name(backend)

        if resolved == "mediapipe":
            self._impl = _MediaPipeBackend()
        elif resolved == "jetson":
            self._impl = _JetsonBackend()
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