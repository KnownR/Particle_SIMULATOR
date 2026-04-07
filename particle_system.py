# ─────────────────────────────────────────
#  particle_system.py  —  Core simulation
# ─────────────────────────────────────────

import numpy as np
import colorsys
import random
from config import *


class ParticleSystem:
    def __init__(self):
        self.max_n = PARTICLE_COUNT_HIGH
        # ── NumPy arrays for vectorised update (fast!) ──
        self.pos  = np.zeros((self.max_n, 2), dtype=np.float32)   # x, y
        self.vel  = np.zeros((self.max_n, 2), dtype=np.float32)   # vx, vy
        self.life = np.zeros(self.max_n,      dtype=np.float32)   # remaining frames
        self.hue  = np.random.rand(self.max_n).astype(np.float32) # for rainbow

        self.active_n   = PARTICLE_COUNT_MID
        self.size       = SIZE_MEDIUM
        self.palette_id = 5          # rainbow default
        self.frozen_colors = None    # set when fist detected
        self.motion_mode  = MODE_CHAOS

        # Physics state
        self.gravity        = GRAVITY_ZERO
        self.attract_force  = ATTRACT_NEUTRAL
        self.vortex_speed   = 0.0    # from wrist tilt

        # Spawn state
        self.spawn_point    = True   # True = point spawn, False = area spawn
        self.pinch_pos      = None   # pixel position of pinch point
        self.drag_particles = []     # indices being dragged by right hand

        # Two-hand state
        self.two_hand_mode  = None   # 'implosion', 'bigbang', 'orbit', None

        # Initialise all particles across screen
        self._init_particles(self.active_n)

    # ────────────────────────────────────
    #  Initialise / respawn
    # ────────────────────────────────────
    def _init_particles(self, n):
        self.pos[:n, 0] = np.random.uniform(0, WIDTH,  n)
        self.pos[:n, 1] = np.random.uniform(0, HEIGHT, n)
        self.vel[:n]    = np.random.uniform(-1, 1, (n, 2))
        self.life[:n]   = np.random.uniform(PARTICLE_LIFETIME // 2,
                                             PARTICLE_LIFETIME, n)

    def _respawn_particle(self, i):
        if self.spawn_point and self.pinch_pos:
            px, py = self.pinch_pos
            self.pos[i] = [px + random.gauss(0, 10),
                           py + random.gauss(0, 10)]
        else:
            self.pos[i] = [random.uniform(0, WIDTH),
                           random.uniform(0, HEIGHT)]
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(0.5, 2.5)
        self.vel[i] = [np.cos(angle) * speed, np.sin(angle) * speed]
        self.life[i] = PARTICLE_LIFETIME
        self.hue[i]  = random.random()

    # ────────────────────────────────────
    #  Apply hand data each frame
    # ────────────────────────────────────
    def apply_left_hand(self, data):
        if data is None:
            return

        px, py = data["palm_norm"]   # 0-1

        # Particle count from Y
        if py < 0.33:
            self.active_n = PARTICLE_COUNT_HIGH
        elif py < 0.66:
            self.active_n = PARTICLE_COUNT_MID
        else:
            self.active_n = PARTICLE_COUNT_LOW

        # Particle size from X
        if px < 0.33:
            self.size = SIZE_SMALL
        elif px < 0.66:
            self.size = SIZE_MEDIUM
        else:
            self.size = SIZE_LARGE

        # Color palette from finger count
        fc = data["finger_count"]
        if fc == 0:
            pass   # fist = freeze colors, don't change palette_id
        else:
            self.palette_id = fc
            self.frozen_colors = None

        # Pinch = spawn mode
        if data["is_pinched"]:
            self.spawn_point = True
            self.pinch_pos   = data["index_tip_px"]
        else:
            self.spawn_point = False
            self.pinch_pos   = None

    def apply_right_hand(self, data):
        if data is None:
            self.gravity       = GRAVITY_ZERO
            self.attract_force = ATTRACT_NEUTRAL
            self.vortex_speed  = 0.0
            return

        px, py = data["palm_norm"]

        # Gravity from Y
        if py < 0.33:
            self.gravity = GRAVITY_STRONG
        elif py < 0.66:
            self.gravity = GRAVITY_ZERO
        else:
            self.gravity = GRAVITY_ANTI

        # Attract/repel from X
        if px < 0.33:
            self.attract_force = ATTRACT_STRONG
        elif px < 0.66:
            self.attract_force = ATTRACT_NEUTRAL
        else:
            self.attract_force = -REPEL_STRONG

        # Motion mode from finger count
        fc = data["finger_count"]
        self.motion_mode = fc   # 0=freeze,1=wind,2=orbital,3=chaos,4=wave,5=boids

        # Vortex from wrist tilt
        tilt = data["tilt_deg"]
        # tilt ≈ -90 when upright; deviation from -90 = rotation drive
        deviation = tilt + 90
        self.vortex_speed = np.clip(deviation / 90.0, -1.0, 1.0) * 0.03

    def apply_two_hand(self, left, right):
        if left is None or right is None:
            self.two_hand_mode = None
            return
        lx, ly = left["palm_norm"]
        rx, ry = right["palm_norm"]
        dist = abs(rx - lx)

        if dist < IMPLOSION_DIST:
            self.two_hand_mode = "implosion"
        elif dist > BIGBANG_DIST:
            self.two_hand_mode = "bigbang"
        else:
            self.two_hand_mode = "orbit"

    # ────────────────────────────────────
    #  Per-frame update
    # ────────────────────────────────────
    def update(self):
        n = self.active_n
        pos = self.pos[:n]
        vel = self.vel[:n]

        # ── Motion mode ──────────────────
        if self.motion_mode == MODE_FREEZE:
            pass   # no velocity update

        elif self.motion_mode == MODE_WIND:
            vel[:, 0] += 0.05   # rightward wind
            vel[:, 1] *= 0.99

        elif self.motion_mode == MODE_ORBITAL:
            cx, cy = WIDTH / 2, HEIGHT / 2
            to_center = np.array([cx, cy]) - pos
            dist = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
            tangent = np.column_stack([-to_center[:, 1], to_center[:, 0]]) / dist
            vel += tangent * 0.08

        elif self.motion_mode == MODE_CHAOS:
            vel += np.random.uniform(-0.3, 0.3, (n, 2))

        elif self.motion_mode == MODE_WAVE:
            t = 0.02
            vel[:, 1] += np.sin(pos[:, 0] * 0.01 + t) * 0.15

        elif self.motion_mode == MODE_BOIDS:
            self._update_boids(pos, vel, n)

        # ── Gravity ──────────────────────
        vel[:, 1] += self.gravity

        # ── Attract / repel from center ──
        if self.attract_force != 0.0:
            cx, cy = WIDTH / 2, HEIGHT / 2
            to_center = np.array([cx, cy]) - pos
            dist = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
            vel += (to_center / dist) * self.attract_force * 0.05

        # ── Vortex (wrist tilt) ──────────
        if abs(self.vortex_speed) > 0.001:
            cx, cy = WIDTH / 2, HEIGHT / 2
            to_center = pos - np.array([cx, cy])
            tangent = np.column_stack([-to_center[:, 1], to_center[:, 0]])
            vel += tangent * self.vortex_speed

        # ── Two-hand effects ─────────────
        if self.two_hand_mode == "implosion":
            cx, cy = WIDTH / 2, HEIGHT / 2
            to_center = np.array([cx, cy]) - pos
            dist = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
            vel += (to_center / dist) * 1.5

        elif self.two_hand_mode == "bigbang":
            cx, cy = WIDTH / 2, HEIGHT / 2
            from_center = pos - np.array([cx, cy])
            dist = np.linalg.norm(from_center, axis=1, keepdims=True) + 1e-6
            vel += (from_center / dist) * 2.0

        elif self.two_hand_mode == "orbit":
            cx, cy = WIDTH / 2, HEIGHT / 2
            to_center = np.array([cx, cy]) - pos
            dist = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
            tangent = np.column_stack([-to_center[:, 1], to_center[:, 0]]) / dist
            vel += tangent * 0.15

        # ── Speed cap ────────────────────
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        too_fast = (speed > 8).flatten()
        vel[too_fast] = vel[too_fast] / speed[too_fast] * 8

        # ── Damping ──────────────────────
        vel *= 0.98

        # ── Move ─────────────────────────
        pos += vel

        # ── Wrap edges ───────────────────
        pos[:, 0] = np.where(pos[:, 0] < 0, WIDTH,
                    np.where(pos[:, 0] > WIDTH, 0, pos[:, 0]))
        pos[:, 1] = np.where(pos[:, 1] < 0, HEIGHT,
                    np.where(pos[:, 1] > HEIGHT, 0, pos[:, 1]))

        # ── Lifetime & respawn ───────────
        self.life[:n] -= 1
        dead = np.where(self.life[:n] <= 0)[0]
        for i in dead:
            self._respawn_particle(i)

    # ── Boids (simplified, n≤500 recommended) ──
    def _update_boids(self, pos, vel, n):
        if n > 500:
            # Only run boids on first 500 to avoid lag
            n = 500
            pos = pos[:n]
            vel = vel[:n]

        for i in range(n):
            diffs = pos - pos[i]
            dists = np.linalg.norm(diffs, axis=1)

            neighbors  = (dists < 50) & (dists > 0)
            near       = (dists < 20) & (dists > 0)

            if neighbors.any():
                # Alignment
                avg_vel = vel[neighbors].mean(axis=0)
                vel[i] += (avg_vel - vel[i]) * 0.05
                # Cohesion
                avg_pos = pos[neighbors].mean(axis=0)
                vel[i] += (avg_pos - pos[i]) * 0.002

            if near.any():
                # Separation
                away = -(pos[near] - pos[i]).mean(axis=0)
                vel[i] += away * 0.1

    # ────────────────────────────────────
    #  Get draw data
    # ────────────────────────────────────
    def get_draw_data(self):
        """Returns list of (x, y, color, size) for active particles."""
        n   = self.active_n
        out = []

        for i in range(n):
            x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
            color = self._get_color(i)
            out.append((x, y, color, self.size))

        return out

    def _get_color(self, i):
        pid = self.palette_id

        if pid == 5:   # Rainbow
            h = float(self.hue[i])
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            return (int(b * 255), int(g * 255), int(r * 255))   # BGR

        palette = PALETTES.get(pid, [(255, 255, 255)])
        rgb = palette[i % len(palette)]
        return (rgb[2], rgb[1], rgb[0])   # RGB → BGR