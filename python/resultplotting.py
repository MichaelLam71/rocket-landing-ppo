import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Get the absolute path of the directory containing THIS script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the paths dynamically
# (Since the JSON and script are in the same folder, look directly in SCRIPT_DIR)
JSON_PATH = os.path.join(SCRIPT_DIR, "results/training_log.json")
SAVE_PATH = os.path.join(SCRIPT_DIR, "training_curves.png")

# Open the file using the absolute path
with open(JSON_PATH) as f:
    log = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Reward ---
ax = axes[0, 0]
ax.plot(log["all_rewards"], alpha=0.2, color='#2196F3')
w = 20
if len(log["all_rewards"]) > w:
    smoothed = np.convolve(log["all_rewards"], np.ones(w)/w, mode='valid')
    ax.plot(smoothed, color='#1565C0', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('PPO Training Reward')
ax.grid(True, alpha=0.3)

# --- Plot 2: Landing rate ---
axes[0, 1].plot(log["iterations"], log["landing_rates"], color='#4CAF50', linewidth=2)
axes[0, 1].set_title('Landing Success Rate')
axes[0, 1].set_ylim(-5, 105)
axes[0, 1].grid(True, alpha=0.3)

# --- Plot 3: Critic Loss ---
axes[1, 0].plot(log["iterations"], log["critic_losses"], color='#F44336')
axes[1, 0].set_title('Critic Loss')
axes[1, 0].grid(True, alpha=0.3)

# --- Plot 4: Entropy ---
axes[1, 1].plot(log["iterations"], log["entropy"], color='#9C27B0')
axes[1, 1].set_title('Entropy')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save using the dynamic path
plt.savefig(SAVE_PATH, dpi=200)
print(f"Plot successfully saved to: {SAVE_PATH}")

plt.show()