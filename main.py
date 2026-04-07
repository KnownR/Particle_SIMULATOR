# ─────────────────────────────────────────
#  main.py  —  Entry point & render loop
# ─────────────────────────────────────────

import cv2
import numpy as np
import time
from config import *
from hand_tracker import HandTracker
from particle_system import ParticleSystem


# ── HUD helper ───────────────────────────
def draw_hud(frame, ps, left, right, fps):
    h, w = frame.shape[:2]

    # Semi-transparent HUD background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)

    # Particle count
    cv2.putText(frame, f"Particles: {ps.active_n}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)

    # Motion mode name
    mode_names = {0: "FREEZE", 1: "WIND", 2: "ORBITAL",
                  3: "CHAOS",  4: "WAVE", 5: "BOIDS"}
    mode_str = mode_names.get(ps.motion_mode, "?")
    cv2.putText(frame, f"Mode: {mode_str}", (w // 2 - 60, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1)

    # Two-hand effect
    if ps.two_hand_mode:
        cv2.putText(frame, f"[{ps.two_hand_mode.upper()}]", (w // 2 - 60, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

    # Palette name
    palette_names = {1: "Mono", 2: "Fire", 3: "Ocean",
                     4: "Aurora", 5: "Rainbow"}
    pal_str = palette_names.get(ps.palette_id, "?")
    cv2.putText(frame, f"Palette: {pal_str}", (w - 180, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 180, 255), 1)

    # Gravity indicator
    g_map = {GRAVITY_STRONG: "↓ Strong", GRAVITY_ZERO: "○ Zero",
             GRAVITY_ANTI: "↑ Anti"}
    g_str = g_map.get(ps.gravity, f"{ps.gravity:.2f}")
    cv2.putText(frame, f"Gravity: {g_str}", (w - 180, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 180, 255), 1)

    # Left/right hand labels
    if left:
        lx, ly = left["palm_px"]
        cv2.putText(frame, "L", (lx - 10, ly - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    if right:
        rx, ry = right["palm_px"]
        cv2.putText(frame, "R", (rx - 10, ry - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 180, 255), 2)


# ── Main loop ────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    tracker = HandTracker()
    ps      = ParticleSystem()

    # Persistent canvas for trails
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    prev_time = time.time()
    print("[INFO] Starting particle simulator. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        frame = cv2.flip(frame, 1)   # Mirror selfie view

        # ── Hand tracking ────────────────
        hands = tracker.process(frame)
        left  = hands["left"]
        right = hands["right"]

        # ── Apply hand data ──────────────
        ps.apply_left_hand(left)
        ps.apply_right_hand(right)
        ps.apply_two_hand(left, right)

        # ── Update simulation ────────────
        ps.update()

        # ── Draw trail fade effect ───────
        fade = np.full_like(canvas, 0)
        cv2.addWeighted(canvas, 1 - TRAIL_ALPHA / 255, fade, TRAIL_ALPHA / 255, 0, canvas)

        # ── Draw particles onto canvas ───
        for x, y, color, size in ps.get_draw_data():
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                cv2.circle(canvas, (x, y), size, color, -1)

        # ── Blend canvas + camera frame ──
        output = cv2.addWeighted(frame, 0.3, canvas, 0.85, 0)

        # ── HUD ──────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_hud(output, ps, left, right, fps)

        cv2.imshow(WINDOW_TITLE, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or ESC
            break
        elif key == ord('r'):              # R = reset canvas
            canvas[:] = 0
            print("[INFO] Canvas reset.")

    # ── Cleanup ──────────────────────────
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()