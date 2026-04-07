# ─────────────────────────────────────────
#  hand_tracker.py  —  MediaPipe wrapper
#  Compatible with mediapipe >= 0.10.x
# ─────────────────────────────────────────

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math
import urllib.request
import os
from config import PINCH_THRESHOLD, WIDTH, HEIGHT

# ── Download the hand landmarker model if not present ──
MODEL_PATH = "hand_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand_landmarker.task (~6 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")

# ── Drawing helper (new API has no auto-draw) ──
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def _draw_landmarks(frame, landmarks_px):
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in CONNECTIONS:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 200, 100), 1)


class HandTracker:
    def __init__(self):
        _ensure_model()

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
        self._ts_ms = 0   # monotonic timestamp for VIDEO mode

    # ── Main process call ────────────────
    def process(self, frame):
        """
        Returns { 'left': HandData | None, 'right': HandData | None }
        """
        self._ts_ms += 33   # ~30 fps cadence — just needs to be monotonic

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        left_data  = None
        right_data = None

        for lm_list, handedness_list in zip(
            result.hand_landmarks,
            result.handedness,
        ):
            # MediaPipe labels are MIRRORED for selfie view — flip them
            raw_label = handedness_list[0].category_name
            label = "Right" if raw_label == "Left" else "Left"

            # Pixel coords for drawing
            landmarks_px = [
                (int(lm.x * w), int(lm.y * h)) for lm in lm_list
            ]
            _draw_landmarks(frame, landmarks_px)

            data = self._extract(lm_list, w, h, landmarks_px)

            if label == "Left":
                left_data = data
            else:
                right_data = data

        return {"left": left_data, "right": right_data}

    # ── Feature extraction ───────────────
    def _extract(self, lm_list, w, h, landmarks_px):
        pts = [(lm.x, lm.y) for lm in lm_list]   # normalised 0-1

        # Palm centre = average of wrist + MCP joints
        palm_x = sum(pts[i][0] for i in [0, 1, 5, 9, 13, 17]) / 6
        palm_y = sum(pts[i][1] for i in [0, 1, 5, 9, 13, 17]) / 6

        fingers_up   = self._count_fingers(pts)
        finger_count = sum(fingers_up)

        pinch_dist = math.dist(pts[4], pts[8])
        is_pinched = pinch_dist < PINCH_THRESHOLD

        # Wrist tilt angle
        wrist   = pts[0]
        mid_mcp = pts[9]
        angle_rad = math.atan2(mid_mcp[1] - wrist[1],
                                mid_mcp[0] - wrist[0])
        tilt_deg = math.degrees(angle_rad)

        fingertips_px = [landmarks_px[i] for i in [4, 8, 12, 16, 20]]

        return {
            "palm_norm":     (palm_x, palm_y),
            "palm_px":       (int(palm_x * w), int(palm_y * h)),
            "finger_count":  finger_count,
            "fingers_up":    fingers_up,
            "is_pinched":    is_pinched,
            "pinch_dist":    pinch_dist,
            "tilt_deg":      tilt_deg,
            "fingertips_px": fingertips_px,
            "index_tip_px":  fingertips_px[1],
        }

    def _count_fingers(self, pts):
        fingers = []
        # Thumb (compare X)
        fingers.append(pts[4][0] < pts[3][0])
        # Index → Pinky (tip above pip joint = finger up)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(pts[tip][1] < pts[pip][1])
        return fingers

    def release(self):
        self.landmarker.close()