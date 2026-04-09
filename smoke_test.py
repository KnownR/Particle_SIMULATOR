import argparse

import numpy as np

from config import HEIGHT, WIDTH
from hand_tracker import HandTracker
from particle_system import ParticleSystem


def run_smoke_test(frames: int, backend: str) -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tracker = HandTracker(backend=backend)
    particle_system = ParticleSystem()

    for _ in range(frames):
        hands = tracker.process(frame)
        left = hands["left"]
        right = hands["right"]

        particle_system.apply_left_hand(left)
        particle_system.apply_right_hand(right)
        particle_system.apply_two_hand(left, right)
        particle_system.update()

        draw_data = particle_system.get_draw_data()
        if not isinstance(draw_data, list):
            raise RuntimeError("Smoke test failed: draw data is not a list")

    tracker.release()
    print(f"[PASS] Smoke test passed for backend='{backend}' over {frames} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless smoke test for Particle Simulator")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to simulate")
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        help="Tracker backend to test: mock|mediapipe|jetson",
    )
    args = parser.parse_args()

    run_smoke_test(frames=args.frames, backend=args.backend)


if __name__ == "__main__":
    main()
