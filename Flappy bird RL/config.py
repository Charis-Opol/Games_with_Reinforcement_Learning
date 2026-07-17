"""
Configuration file
"""

# ===========================
# Window
# ===========================

WIDTH = 500
HEIGHT = 700

FPS = 60

# ===========================
# Bird
# ===========================

BIRD_RADIUS = 20

GRAVITY = 0.65

JUMP_STRENGTH = -9

MAX_FALL_SPEED = 10

# Minimum number of frames between jumps (prevents jump spamming)
JUMP_COOLDOWN_FRAMES = 5

ROTATION_SPEED = 3

MAX_UP_ROTATION = 25

MAX_DOWN_ROTATION = -90

# ===========================
# Colours
# ===========================

WHITE = (255,255,255)

BLACK = (0,0,0)

BLUE = (135,206,235)

YELLOW = (255,220,0)

ORANGE = (255,140,0)

# ===========================
# Pipes
# ===========================

PIPE_WIDTH = 80
PIPE_GAP = 180
PIPE_SPEED = 4
PIPE_MIN_HEIGHT = 80
PIPE_MAX_HEIGHT = 450

PIPE_COLOR = (34, 177, 76)

# ===========================
# Ground
# ===========================

GROUND_HEIGHT = 100
GROUND_SPEED = PIPE_SPEED
GROUND_COLOR = (160, 82, 45)

# ===========================
# Text
# ===========================

FONT_SIZE = 36

TEXT_COLOR = (255, 255, 255)

# ===========================
# Assistance & shaping
# ===========================
# Enable a simple rule-based assistant that can override/guide actions
ASSISTED_MODE = False
# Probability (0-1) that the assistant will be applied when enabled
ASSIST_PROBABILITY = 0.2
# Margin in pixels around gap center to trigger a corrective jump
ASSIST_MARGIN = 15
# Reward shaping weight (added fraction when bird is closer to gap center)
SHAPING_WEIGHT = 0.4
# Separate shaping weights for vertical and horizontal proximity
SHAPING_VERTICAL_WEIGHT = 0.4
SHAPING_HORIZONTAL_WEIGHT = 0.2