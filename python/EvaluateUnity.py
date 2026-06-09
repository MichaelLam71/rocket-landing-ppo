import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import socket
import struct
import math
import time
from config import OBS_SIZE, ACTION_SIZE, HIDDEN, HOST, PORT, POS_SCALE, VEL_SCALE

SAVE_DIR = os.path.join(os.path.dirname(__file__), "NN")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_EPISODES = 10000
TO_LAND_SCALE = 50.0


class UnityEnv:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print("Waiting for Unity to connect...")
        self.conn, _ = self.server.accept()
        print("Unity connected!")

    def _recv_floats(self, n):
        data = b''
        while len(data) < n * 4:
            chunk = self.conn.recv(n * 4 - len(data))
            if chunk == b'':
                raise ConnectionError("Unity disconnected")
            data += chunk
        return struct.unpack(f'{n}f', data)

    def _send_floats(self, values):
        self.conn.sendall(struct.pack(f'{len(values)}f', *values))

    def reset(self):
        self._send_floats([-999.0, 0.0, 0.0])
        response = self._recv_floats(OBS_SIZE + 2)
        return np.array(response[:OBS_SIZE])

    def step(self, action):
        self._send_floats(action.tolist())
        response = self._recv_floats(OBS_SIZE + 2)
        obs = np.array(response[:OBS_SIZE])
        reward = response[OBS_SIZE]
        done = bool(response[OBS_SIZE + 1])
        return obs, reward, done

    def close(self):
        self.conn.close()
        self.server.close()


class Actor(nn.Module):
    def __init__(self, inp, hidden, outp):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, outp)
        self.relu = nn.ReLU()
        self.log_std = nn.Parameter(torch.zeros(outp))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.fc3(x)
        stds = self.log_std.exp()
        return torch.distributions.Normal(mean, stds)


def decode_obs(obs):
    height = obs[1] * POS_SCALE
    vel = np.array([obs[3], obs[4], obs[5]]) * VEL_SCALE
    speed = np.linalg.norm(vel)
    up_y = max(-1.0, min(1.0, obs[7]))
    tilt = math.degrees(math.acos(up_y))
    to_pad = np.array([obs[12], obs[13], obs[14]]) * TO_LAND_SCALE
    dist = np.linalg.norm(to_pad)
    return height, speed, tilt, dist


# load trained actor
actor = Actor(OBS_SIZE, HIDDEN, ACTION_SIZE)
actor.load_state_dict(torch.load(f"{SAVE_DIR}/actor.pth"))
actor.eval()

env = UnityEnv()

successes = 0
landing_speeds = []
landing_tilts = []
landing_dists = []
episode_lengths = []
episode_rewards = []

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0
    last_obs = obs

    while not done:
        with torch.no_grad():
            dist = actor(torch.tensor(obs, dtype=torch.float32))
            action = dist.mean.clamp(-1, 1)

        obs, reward, done = env.step(action.numpy())
        total_reward += reward
        last_obs = obs
        steps += 1

    height, speed, tilt, dist_to_pad = decode_obs(last_obs)
    landed = total_reward > 0

    if landed:
        successes += 1
        landing_speeds.append(speed)
        landing_tilts.append(tilt)
        landing_dists.append(dist_to_pad)

    episode_lengths.append(steps)
    episode_rewards.append(total_reward)

    status = "LANDED" if landed else "crashed"
    print(f"Episode {ep+1:2d}: {status:8s} | "
          f"speed={speed:5.2f} m/s | tilt={tilt:5.1f} deg | "
          f"dist={dist_to_pad:5.2f} m | reward={total_reward:7.1f}")

    time.sleep(1.0)

env.close()

success_rate = 100.0 * successes / NUM_EPISODES

print("\n" + "=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)
print(f"Episodes evaluated:        {NUM_EPISODES}")
print(f"Successful landings:       {successes}")
print(f"Success rate:              {success_rate:.1f}%")
print(f"Mean episode reward:       {np.mean(episode_rewards):.1f}")
print(f"Mean episode length:       {np.mean(episode_lengths):.0f} steps")
if landing_speeds:
    print(f"Mean landing speed:        {np.mean(landing_speeds):.2f} m/s")
    print(f"Mean landing tilt:         {np.mean(landing_tilts):.1f} deg")
    print(f"Mean dist from pad:        {np.mean(landing_dists):.2f} m")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].pie(
    [successes, NUM_EPISODES - successes],
    labels=["Landed", "Crashed"],
    autopct="%1.0f%%",
    colors=["#4CAF50", "#F44336"]
)
axes[0, 0].set_title(f"Landing Success Rate ({success_rate:.0f}%)")

axes[0, 1].plot(episode_rewards, marker="o", markersize=3)
axes[0, 1].set_title("Reward per Episode")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Total Reward")

if landing_speeds:
    axes[1, 0].hist(landing_speeds, bins=10, color="#2196F3", edgecolor="black")
    axes[1, 0].set_title("Landing Speed Distribution")
    axes[1, 0].set_xlabel("Speed (m/s)")
    axes[1, 0].set_ylabel("Count")

if landing_tilts:
    axes[1, 1].hist(landing_tilts, bins=10, color="#FF9800", edgecolor="black")
    axes[1, 1].set_title("Landing Tilt Distribution")
    axes[1, 1].set_xlabel("Tilt (degrees)")
    axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/evaluation_stats.png", dpi=150)
print(f"\nSaved stats plot to {RESULTS_DIR}/evaluation_stats.png")
plt.show()