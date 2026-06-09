"""
PID_Collect.py
--------------
Connects to Unity through the same socket bridge as the PPO script.
Runs a simple PID controller that lands the rocket, and records
(observation, action) pairs as training data for behavioral cloning.
Only saves successful landing episodes.
"""

import numpy as np
import socket
import struct
import os


# ==================== CONFIG ====================
from config import *

NUM_EPISODES = 100
SAVE_PATH = os.path.join(os.path.dirname(__file__), "demos.npz")


# ==================== SOCKET ====================
class UnityEnv:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print("Waiting for Unity...")
        self.conn, _ = self.server.accept()
        print("Unity connected!")

    def _recv(self, n):
        data = b''
        while len(data) < n * 4:
            chunk = self.conn.recv(n * 4 - len(data))
            if chunk == b'': raise ConnectionError("Unity disconnected")
            data += chunk
        return struct.unpack(f'{n}f', data)

    def _send(self, values):
        self.conn.sendall(struct.pack(f'{len(values)}f', *values))

    def reset(self):
        self._send([-999.0, 0.0, 0.0])
        resp = self._recv(OBS_SIZE + 2)
        return np.array(resp[:OBS_SIZE])

    def step(self, action):
        self._send(action)
        resp = self._recv(OBS_SIZE + 2)
        obs = np.array(resp[:OBS_SIZE])
        reward = resp[OBS_SIZE]
        done = bool(resp[OBS_SIZE + 1])
        return obs, reward, done

    def close(self):
        self.conn.close()
        self.server.close()


# ==================== PID CONTROLLER ====================
def pid_action(obs):
    height = obs[1] * POS_SCALE
    vel_y  = obs[4] * VEL_SCALE
    height_above_pad = max(height - PAD_HEIGHT, 0.1)

    tilt_x = obs[6]
    tilt_z = obs[8]
    ang_vel_x = obs[9] * ANG_VEL_SCALE
    ang_vel_z = obs[11] * ANG_VEL_SCALE

    # vertical thrust (suicide burn)
    if vel_y > 0:
        thrust_frac = 0.0
    else:
        speed = abs(vel_y)

        if height_above_pad < 0.3 and speed < 1.0:
            return [-1.0, 0.0, 0.0]

        a_max = MAX_THRUST / MASS - GRAVITY
        brake_dist = (speed * speed) / (2.0 * a_max)
        margin = max(2.0, speed * 0.15)

        if brake_dist + margin >= height_above_pad:
            required_decel = (speed * speed) / (2.0 * height_above_pad)
            thrust_frac = (required_decel + GRAVITY) * MASS / MAX_THRUST
            thrust_frac = min(thrust_frac, 1.0)
        else:
            thrust_frac = 0.0

    action_thrust = thrust_frac * 2.0 - 1.0

    # gimbal PD
    gimbal_x = np.clip(-KP_GIMBAL * tilt_z + KD_GIMBAL * ang_vel_x, -1, 1)
    gimbal_z = np.clip(KP_GIMBAL * tilt_x - KD_GIMBAL * ang_vel_z, -1, 1)

    return [action_thrust, gimbal_x, gimbal_z]


# ==================== COLLECT DATA ====================
env = UnityEnv()

all_obs = []
all_actions = []
landings = 0
crashes = 0

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    steps_count = 0

    episode_obs = []
    episode_actions = []

    while not done:
        action = pid_action(obs)
        
        height = obs[1] * POS_SCALE
        vel_y = obs[4] * VEL_SCALE
        if ep == 0:
            print(f"  step={steps_count} h={height:.2f} vy={vel_y:.2f} "
                  f"thrust={action[0]:.3f} gX={action[1]:.3f} gZ={action[2]:.3f} "
                  f"tiltX={obs[6]:.4f} tiltZ={obs[8]:.4f} "
                  f"angX={obs[9]*ANG_VEL_SCALE:.3f} angZ={obs[11]*ANG_VEL_SCALE:.3f}")
        steps_count += 1
        
        episode_obs.append(obs.copy())
        episode_actions.append(action)
        obs, reward, done = env.step(action)

    if reward > 0:
        landings += 1
        all_obs.extend(episode_obs)
        all_actions.extend(episode_actions)
    else:
        crashes += 1
        height = obs[1] * POS_SCALE
        vel_y = obs[4] * VEL_SCALE
        tilt_x = obs[6]
        tilt_z = obs[8]
        print(f"  CRASH ep={ep+1} h={height:.1f} vy={vel_y:.1f} "
              f"tiltX={tilt_x:.3f} tiltZ={tilt_z:.3f}")

    if (ep + 1) % 20 == 0:
        obs_array = np.array(all_obs, dtype=np.float32)
        act_array = np.array(all_actions, dtype=np.float32)
        np.savez(SAVE_PATH, observations=obs_array, actions=act_array)
        rate = 100 * landings / (ep + 1)
        print(f"Episode {ep+1}/{NUM_EPISODES}  "
              f"landings={landings}  crashes={crashes}  "
              f"samples={len(all_obs)}  saved!  rate={rate:.0f}%")

env.close()

obs_array = np.array(all_obs, dtype=np.float32)
act_array = np.array(all_actions, dtype=np.float32)
np.savez(SAVE_PATH, observations=obs_array, actions=act_array)

print(f"\nDone! Saved {len(all_obs)} successful samples to {SAVE_PATH}")
print(f"Landing rate: {100*landings/NUM_EPISODES:.0f}%")