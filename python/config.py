"""
config.py
---------
Single source of truth for all physics and environment constants.
Change values here, not in individual scripts.
Must match Unity Inspector values.
"""

# Rocket physics
DRY_MASS = 22000.0
FUEL_MASS = 2000.0
MASS = DRY_MASS + FUEL_MASS          # 24000 kg
TWR = 2.0
MAX_THRUST = MASS * 9.81 * TWR       # ~400248 N
GRAVITY = 9.81

# Environment
SPAWN_HEIGHT = 200.0
PAD_HEIGHT = 5.2

# Observation scaling (must match PythonBridge Inspector)
POS_SCALE = 50.0
VEL_SCALE = 20.0
ANG_VEL_SCALE = 10.0

# PID tuning
KP_GIMBAL = 0.8
KD_GIMBAL = 0.0

# Network
OBS_SIZE = 15
ACTION_SIZE = 3
HIDDEN = 256

# Socket
HOST = "127.0.0.1"
PORT = 5005



# ==================== AUTO-EXPORT TO UNITY ====================
import json
import os

config_dict = {
    "DRY_MASS": DRY_MASS,
    "FUEL_MASS": FUEL_MASS,
    "MAX_THRUST": MAX_THRUST,
    "GRAVITY": GRAVITY,
    "PAD_HEIGHT": PAD_HEIGHT,
    "POS_SCALE": POS_SCALE,
    "VEL_SCALE": VEL_SCALE,
    "ANG_VEL_SCALE": ANG_VEL_SCALE
}

# Saves 'config.json' in the same folder as your python scripts
json_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(json_path, "w") as f:
    json.dump(config_dict, f, indent=4)