"""
BehavioralCloning.py
--------------------
Trains the Actor network to imitate the PID controller using supervised learning.

Usage:
  1. Run PID_Collect.py first to generate demos.npz
  2. Run this script: python Behaviorcloning.py
  3. It trains the Actor and saves weights to NN/actor.pth
  4. Use EvaluateUnity.py or PPO with RESUME=True to test/fine-tune
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from config import OBS_SIZE, ACTION_SIZE, HIDDEN

DEMO_PATH = os.path.join(os.path.dirname(__file__), "demos.npz")
SAVE_DIR  = os.path.join(os.path.dirname(__file__), "NN")
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001


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

    def predict(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# load demos
print("Loading demos...")
data = np.load(DEMO_PATH)
obs_data = torch.tensor(data['observations'], dtype=torch.float32)
act_data = torch.tensor(data['actions'], dtype=torch.float32)
print(f"Loaded {len(obs_data)} samples")

# train/val split (90/10)
n = len(obs_data)
perm = torch.randperm(n)
split = int(0.9 * n)
train_obs, val_obs = obs_data[perm[:split]], obs_data[perm[split:]]
train_act, val_act = act_data[perm[:split]], act_data[perm[split:]]

# create network and optimizer
actor = Actor(OBS_SIZE, HIDDEN, ACTION_SIZE)
optimizer = torch.optim.Adam(actor.parameters(), lr=LR)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []

print(f"Training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    perm = torch.randperm(len(train_obs))
    train_obs = train_obs[perm]
    train_act = train_act[perm]

    epoch_loss = 0
    n_batches = 0
    for i in range(0, len(train_obs), BATCH_SIZE):
        batch_obs = train_obs[i:i+BATCH_SIZE]
        batch_act = train_act[i:i+BATCH_SIZE]

        predicted = actor.predict(batch_obs)
        loss = loss_fn(predicted, batch_act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    with torch.no_grad():
        val_pred = actor.predict(val_obs)
        val_loss = loss_fn(val_pred, val_act).item()

    avg_train = epoch_loss / n_batches
    train_losses.append(avg_train)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}  train_loss={avg_train:.6f}  val_loss={val_loss:.6f}")

# save with tightened exploration noise
actor.log_std.data.fill_(-3.0)
torch.save(actor.state_dict(), f"{SAVE_DIR}/actor.pth")
print(f"\nSaved actor to {SAVE_DIR}/actor.pth")

# plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Behavioral Cloning Training")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "results", "bc_training.png"), dpi=150)
plt.show()