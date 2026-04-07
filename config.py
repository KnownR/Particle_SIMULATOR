# ─────────────────────────────────────────
#  config.py  —  All tunable constants
# ─────────────────────────────────────────

# Window
WIDTH, HEIGHT = 1280, 720
FPS = 60
WINDOW_TITLE = "Particle Simulator — Hand Controlled"

# ── Particle limits ──────────────────────
PARTICLE_COUNT_LOW    = 100
PARTICLE_COUNT_MID    = 1000
PARTICLE_COUNT_HIGH   = 3000   # capped at 3000 for Windows perf (5000 on Jetson later)

# ── Particle appearance ──────────────────
SIZE_SMALL  = 2
SIZE_MEDIUM = 4
SIZE_LARGE  = 8

# ── Physics defaults ─────────────────────
GRAVITY_STRONG   =  0.4   # downward
GRAVITY_ZERO     =  0.0
GRAVITY_ANTI     = -0.3   # upward

ATTRACT_STRONG   = 0.8    # black hole
ATTRACT_NEUTRAL  = 0.0
REPEL_STRONG     = 0.8    # explosion

# ── Motion modes (right hand finger count) ──
MODE_FREEZE  = 0
MODE_WIND    = 1
MODE_ORBITAL = 2
MODE_CHAOS   = 3
MODE_WAVE    = 4
MODE_BOIDS   = 5          # bonus — may be slow

# ── Color palettes ───────────────────────
PALETTES = {
    1: [(255, 255, 255)],                                           # Monochrome
    2: [(255, 60, 0), (255, 140, 0), (255, 220, 0)],               # Fire
    3: [(0, 120, 255), (0, 220, 255), (0, 200, 180)],              # Ocean
    4: [(80, 255, 120), (180, 0, 255), (255, 80, 200)],            # Aurora
    5: None,   # Rainbow — computed per-particle via HSV
}

# ── Pinch threshold ──────────────────────
PINCH_THRESHOLD = 0.07   # normalised distance between thumb tip & index tip

# ── Two-hand distance thresholds (fraction of screen width) ──
IMPLOSION_DIST = 0.15
BIGBANG_DIST   = 0.70

# ── Lifetime / fade ──────────────────────
PARTICLE_LIFETIME = 180   # frames  (3 s @ 60 fps)

# ── Trail ────────────────────────────────
TRAIL_ALPHA = 40          # 0-255 — lower = longer trails