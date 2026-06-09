import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader
import socket
import struct
import json
from config import OBS_SIZE, ACTION_SIZE, HIDDEN, HOST, PORT

SAVE_DIR = os.path.join(os.path.dirname(__file__), "NN")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class UnityEnv:
    def __init__(self):
        self.obs_size = OBS_SIZE
        self.action_size = ACTION_SIZE
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
        response = self._recv_floats(self.obs_size + 2)
        return np.array(response[:self.obs_size]), {}

    def step(self, action):
        self._send_floats(action.tolist())
        response = self._recv_floats(self.obs_size + 2)
        obs = np.array(response[:self.obs_size])
        reward = response[self.obs_size]
        done = bool(response[self.obs_size + 1])
        return obs, reward, done, False, {}

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


class Critic(nn.Module):
    def __init__(self, inp, hidden, outp):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, outp)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class Buffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []


# Hyperparameters
actor_lr = 0.00003
critic_lr = 0.0001
steps = 2048
gamma = 0.99
gae_lambda = 0.95
num_epochs = 2
num_iterations = 6000
best_avg = -np.inf

# Setup
env = UnityEnv()
actor = Actor(OBS_SIZE, HIDDEN, ACTION_SIZE)
critic = Critic(OBS_SIZE, HIDDEN, 1)
buffer = Buffer()

actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

RESUME = True

if RESUME and os.path.exists(f"{SAVE_DIR}/actor.pth"):
    actor.load_state_dict(torch.load(f"{SAVE_DIR}/actor.pth"))
    actor.log_std.data.fill_(-2.0) 
    if os.path.exists(f"{SAVE_DIR}/critic.pth"):
        critic.load_state_dict(torch.load(f"{SAVE_DIR}/critic.pth"))
    print("Loaded saved weights!")
else:
    print("Starting fresh.")

best_avg = -np.inf
reward_history = []

# ==================== TRACKING ====================
log_iterations = []
log_avg_rewards = []
log_actor_losses = []
log_critic_losses = []
log_landing_rates = []
log_entropy = []

# Training loop
for iteration in range(num_iterations):
    step_count = 0

    lr = actor_lr * (1 - iteration / num_iterations)
    for pg in actor_optim.param_groups:
        pg['lr'] = lr
    for pg in critic_optim.param_groups:
        pg['lr'] = lr

    # collect experience
    while step_count < steps:
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done and step_count < steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = actor(obs_tensor)
                sample = dist.sample()
                log_prob = dist.log_prob(sample).sum()
                value = critic(obs_tensor).squeeze()
                action = sample.clamp(-1, 1)
            

            next_obs, reward, terminated, truncated, _ = env.step(
                action.detach().numpy()
            )
            done = terminated or truncated

            buffer.store(obs, sample, log_prob, reward, value, done)

            obs = next_obs
            episode_reward += reward
            step_count += 1

        if done:
            reward_history.append(episode_reward)

    # compute GAE advantages
    advantages = []
    gae = 0
    for i in reversed(range(len(buffer.states))):
        if buffer.dones[i]:
            future_value = 0
            gae = 0
        elif i == len(buffer.states) - 1:
            future_value = critic(
                torch.tensor(buffer.states[-1], dtype=torch.float32)
            ).squeeze().detach()
        else:
            future_value = buffer.values[i + 1].detach()

        delta = buffer.rewards[i] + gamma * future_value - buffer.values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages).detach()
    values = torch.stack(buffer.values).detach()
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
    actions = torch.stack(buffer.actions).detach()
    old_log_probs = torch.stack(buffer.log_probs).detach()

    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # PPO update
    epoch_actor_loss = 0
    epoch_critic_loss = 0
    epoch_entropy = 0
    n_updates = 0

    for epoch in range(num_epochs):
        for bs, ba, blp, badv, bret in loader:
            new_dist = actor(bs)
            new_log_prob = new_dist.log_prob(ba).sum(-1)
            ratio = torch.exp(new_log_prob - blp)
            clipped = torch.clamp(ratio, 0.95, 1.05)
            actor_loss = -torch.min(ratio * badv, clipped * badv).mean()

            entropy = new_dist.entropy().mean()
            actor_loss -= 0.01 * entropy

            new_value = critic(bs).squeeze()
            critic_loss = ((new_value - bret) ** 2).mean()

            actor_optim.zero_grad()
            critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            actor_optim.step()
            critic_optim.step()

            with torch.no_grad():
                actor.log_std.data.clamp_(min=-2.5)

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()
            epoch_entropy += entropy.item()
            n_updates += 1

    buffer.clear()

    # logging
    if iteration % 10 == 0:
        avg = np.mean(reward_history[-10:]) if reward_history else 0
        recent = reward_history[-10:] if reward_history else []
        landing_rate = sum(1 for r in recent if r > 0) / max(len(recent), 1) * 100
        avg_a_loss = epoch_actor_loss / max(n_updates, 1)
        avg_c_loss = epoch_critic_loss / max(n_updates, 1)
        avg_ent = epoch_entropy / max(n_updates, 1)

        log_iterations.append(iteration)
        log_avg_rewards.append(avg)
        log_actor_losses.append(avg_a_loss)
        log_critic_losses.append(avg_c_loss)
        log_landing_rates.append(landing_rate)
        log_entropy.append(avg_ent)

        print(f"\nIteration {iteration}")
        print(f"  Avg Reward: {avg:.2f}  |  Landing Rate: {landing_rate:.0f}%")
        print(f"  Actor Loss: {avg_a_loss:.4f}  |  Critic Loss: {avg_c_loss:.4f}")
        print(f"  Entropy: {avg_ent:.4f}")

        if avg > best_avg:
            best_avg = avg
            torch.save(actor.state_dict(), f"{SAVE_DIR}/actor.pth")
            torch.save(critic.state_dict(), f"{SAVE_DIR}/critic.pth")
            print(f"  ** New best model saved! **")

        # save training log every 50 iterations (recoverable if crash)
        if iteration % 50 == 0:
            log_data = {
                "iterations": log_iterations,
                "avg_rewards": log_avg_rewards,
                "actor_losses": log_actor_losses,
                "critic_losses": log_critic_losses,
                "landing_rates": log_landing_rates,
                "entropy": log_entropy,
                "all_rewards": reward_history
            }
            with open(f"{RESULTS_DIR}/training_log.json", "w") as f:
                json.dump(log_data, f)


# ==================== FINAL PLOTS ====================
def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Reward curve (per episode, smoothed)
ax = axes[0, 0]
ax.plot(reward_history, alpha=0.2, color='#2196F3', label='Raw')
ax.plot(smooth(reward_history, 20), color='#1565C0', linewidth=2, label='Smoothed (20)')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('PPO Training Reward')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Landing rate over training
ax = axes[0, 1]
ax.plot(log_iterations, log_landing_rates, color='#4CAF50', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Landing Rate (%)')
ax.set_title('Landing Success Rate')
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

# 3. Actor and Critic loss
ax = axes[1, 0]
ax.plot(log_iterations, log_actor_losses, color='#FF9800', linewidth=1.5, label='Actor')
ax.set_xlabel('Iteration')
ax.set_ylabel('Actor Loss')
ax.set_title('Training Losses')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax2 = ax.twinx()
ax2.plot(log_iterations, log_critic_losses, color='#F44336', linewidth=1.5, label='Critic')
ax2.set_ylabel('Critic Loss')
ax2.legend(loc='upper right')

# 4. Entropy
ax = axes[1, 1]
ax.plot(log_iterations, log_entropy, color='#9C27B0', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Entropy')
ax.set_title('Policy Entropy')
ax.grid(True, alpha=0.3)

plt.suptitle('PPO Rocket Landing Training', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=200, bbox_inches='tight')
print(f"\nSaved training curves to {RESULTS_DIR}/training_curves.png")
plt.show()

# save final log
log_data = {
    "iterations": log_iterations,
    "avg_rewards": log_avg_rewards,
    "actor_losses": log_actor_losses,
    "critic_losses": log_critic_losses,
    "landing_rates": log_landing_rates,
    "entropy": log_entropy,
    "all_rewards": reward_history
}
with open(f"{RESULTS_DIR}/training_log.json", "w") as f:
    json.dump(log_data, f)
print(f"Saved training log to {RESULTS_DIR}/training_log.json")