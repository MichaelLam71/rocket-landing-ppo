import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader
import socket
import struct

SAVE_DIR = os.path.join(os.path.dirname(__file__), "NN")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class UnityEnv:
    def __init__(self, host="127.0.0.1", port=5005, obs_size=15, action_size=3):
        self.obs_size = obs_size
        self.action_size = action_size
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
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
actor_lr = 0.0001
critic_lr = 0.0001
steps = 512
gamma = 0.99
gae_lambda = 0.95
num_epochs = 2
num_iterations = 6000
best_avg = -np.inf

# Setup
env = UnityEnv()
actor = Actor(env.obs_size, 256, env.action_size)
critic = Critic(env.obs_size, 256, 1)
buffer = Buffer()

actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

# load pretrained weights for curriculum learning (set to False for fresh start)
RESUME = True


if RESUME and os.path.exists(f"{SAVE_DIR}/actor.pth"):
    actor.load_state_dict(torch.load(f"{SAVE_DIR}/actor.pth"))
    if os.path.exists(f"{SAVE_DIR}/critic.pth"):
        critic.load_state_dict(torch.load(f"{SAVE_DIR}/critic.pth"))
    print("Loaded saved weights!")
else:
    print("Starting fresh.")

# reset tracking for the new stage
best_avg = -np.inf
reward_history = []

reward_history = []

# Training loop
for iteration in range(num_iterations):
    step_count = 0

    # learning rate decay
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
            dist = actor(obs_tensor)
            sample = dist.sample()
            action = sample.clamp(-1, 1)
            log_prob = dist.log_prob(sample).sum()
            value = critic(obs_tensor).squeeze()

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

    # compute returns before normalizing advantages
    advantages = torch.stack(advantages).detach()
    values = torch.stack(buffer.values).detach()
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # prepare tensors
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
    actions = torch.stack(buffer.actions).detach()
    old_log_probs = torch.stack(buffer.log_probs).detach()

    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # PPO update
    for epoch in range(num_epochs):
        for bs, ba, blp, badv, bret in loader:
            # actor loss (clipped surrogate objective)
            new_dist = actor(bs)
            new_log_prob = new_dist.log_prob(ba).sum(-1)
            ratio = torch.exp(new_log_prob - blp)
            clipped = torch.clamp(ratio, 0.95, 1.05)
            actor_loss = -torch.min(ratio * badv, clipped * badv).mean()

            # entropy bonus
            entropy = new_dist.entropy().mean()
            actor_loss -= 0.01 * entropy

            # critic loss
            new_value = critic(bs).squeeze()
            critic_loss = ((new_value - bret) ** 2).mean()

            # backprop
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            actor_optim.step()
            critic_optim.step()

    buffer.clear()

    # logging
    if iteration % 10 == 0:
        avg = np.mean(reward_history[-10:]) if reward_history else 0
        print(f"\nIteration {iteration}")
        print(f"  Avg Reward: {avg:.2f}")
        print(f"  Actor Loss: {actor_loss.item():.4f}")
        print(f"  Critic Loss: {critic_loss.item():.4f}")

        if avg > best_avg:
            best_avg = avg
            torch.save(actor.state_dict(), f"{SAVE_DIR}/actor.pth")
            torch.save(critic.state_dict(), f"{SAVE_DIR}/critic.pth")

# plot training curve
window = 20
smoothed = np.convolve(reward_history, np.ones(window) / window, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Unity Rocket PPO")
plt.savefig(f"{RESULTS_DIR}/training_curve.png")
plt.show()