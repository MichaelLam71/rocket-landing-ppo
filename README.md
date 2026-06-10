# Rocket Landing with Reinforcement Learning

A rocket landing simulation using Proximal Policy Optimization (PPO) implemented from scratch in Unity 3D, with realistic fuel consumption and variable mass. The agent learns to perform a suicide burn descent and soft vertical landing using a two-phase training pipeline: Behavioral Cloning (BC) from a PID controller, followed by PPO fine-tuning.

## Architecture

Python handles all machine learning (PyTorch). Unity handles all physics simulation. They communicate over a TCP socket on port 5005. Python is the server, Unity is the client.

```
Python (PPO / BC / PID)  <---TCP socket--->  Unity (Physics / Rendering)
         |                                          |
    Neural Network                           RocketController.cs
    Reward logic                             PythonBridge.cs
    Training loop                            Rigidbody physics
```
## Socket Protocol

Python sends 3 floats (thrust, rcsX, rcsZ). Unity replies with 17 floats (15 observations + 1 reward + 1 done flag). Reset signal: thrust = -999.

## Project Structure

```
├── Assets/                    # Unity project files
│   └── Scripts/
│       ├── RocketController.cs
│       └── PythonBridge.cs
├── ProjectSettings/           # Unity project settings
├── Packages/                  # Unity package manifest
├── python/
│   ├── config.py              # Single source of truth for all constants
│   ├── PID_collect.py         # Step 1: PID controller collects landing demonstrations
│   ├── Behaviorcloning.py     # Step 2: Train neural network to imitate PID
│   ├── Unity_PPO.py           # Step 3: Fine-tune with PPO reinforcement learning
│   ├── EvaluateUnity.py       # Step 4: Test trained policy (deterministic)
│   ├── demos.npz              # Generated: PID demonstration data
│   ├── NN/
│   │   ├── actor.pth          # Trained actor (policy) weights
│   │   └── critic.pth         # Trained critic (value function) weights
│   └── results/
│       ├── bc_training.png    # BC loss curves
│       ├── training_curves.png # PPO training curves
│       └── training_log.json  # PPO training log (recoverable)
└── .gitignore
```

## How It Works

### The Two-Phase Training Pipeline

**Phase 1 -- Behavioral Cloning (supervised learning)**

A PID controller lands the rocket using a suicide burn algorithm. Every (observation, action) pair is recorded as training data, and a neural network is trained to imitate the PID via supervised learning (MSE loss). This gives the network a competent starting policy without any reward engineering. Only successful landing episodes are saved to the training data.

**Phase 2 -- PPO Fine-Tuning (reinforcement learning)**

The BC-trained network is loaded into PPO, which fine-tunes it using terminal-only rewards (+100 to +300 for landing, -100 for crashing). PPO improves landing quality (softer, more upright) and generalises to harder conditions (more tilt, position offset, initial velocity).

### Why Behavioural Cloning?

Training PPO without BC pre-training was attempted, but the agent consistently exploited reward function loopholes rather than learning to land. Dense reward shaping (rewarding the agent for getting closer to the pad) caused two failure modes: hovering (the agent farms small per-step rewards by staying alive) and slamming (at high simulation speed, moving faster toward the pad yields more shaping reward per step). Terminal-only rewards are too sparse for a random policy to discover landing by chance. BC solves this by giving PPO a working starting point.

### Observation Space (15 floats)

| Index | Value | Scaling |
|-------|-------|---------|
| 0-2 | Position (x, y, z) | / 50 |
| 3-5 | Velocity (x, y, z) | / 20 |
| 6-8 | Up vector (x, y, z) | raw (encodes tilt) |
| 9-11 | Angular velocity (x, y, z) | / 10 |
| 12-14 | Vector to landing pad (x, y, z) | / 50 |

All values clamped to [-5, 5].

### Action Space (3 floats, continuous)

| Index | Value | Range | Mapping |
|-------|-------|-------|---------|
| 0 | Main engine thrust | [-1, 1] | mapped to [0, 1] in Unity |
| 1 | RCS torque X-axis | [-1, 1] | controls pitch |
| 2 | RCS torque Z-axis | [-1, 1] | controls roll |

### Physics Model

The simulation includes fuel consumption and aerodynamic drag. Fuel depletes based on specific impulse (mass flow rate = thrust / (Isp * g0)), reducing total mass over the descent. The engine cuts out if fuel reaches zero. Aerodynamic drag follows the standard drag equation (F = 0.5 * rho * Cd * A * v^2), opposing the velocity vector. Both effects reset at the start of each episode. The inertia tensor is manually set to a symmetric value to eliminate cross-axis coupling from the asymmetric rocket model geometry.

### Reward Function (terminal only)
 
- **Successful landing:** 100 + speed_bonus + tilt_bonus + proximity_bonus (max 300)
  - speed_bonus = (1 - speed / speed_limit) * 100
  - tilt_bonus = (1 - tilt / tilt_limit) * 50
  - proximity_bonus = (1 - distance_to_pad / landing_radius) * 50
- **Crash:** -100 (includes landing too fast, too tilted, or too far from pad)
- **Per step:** -0.01 (small time penalty)
No distance-based shaping during flight. This is intentional and critical for stability at high simulation speeds. The proximity bonus only applies at touchdown.

### PPO Algorithm

PPO is a policy gradient algorithm using two neural networks. The **Actor** takes the current state and outputs a Gaussian distribution over the 3 continuous actions. Actions are sampled during training for exploration, and the distribution mean is used during evaluation. The **Critic** takes the same state and outputs a value estimate, predicting total expected future reward from that state.

Each iteration collects 2048 steps of experience, then computes advantages using Generalized Advantage Estimation (GAE, lambda=0.95), answering: "was this action better or worse than expected?" The policy is then updated using the clipped surrogate objective, where the probability ratio between new and old policy is clamped to [0.95, 1.05] to prevent destructively large updates.

Additional stability mechanisms include an entropy bonus (prevents the policy from collapsing to deterministic actions), gradient clipping (caps update magnitude from outlier batches), learning rate linear decay, action clamping to [-1, 1], and a log_std floor (clamped at -2.5 to maintain minimum exploration).

### Network Architecture

- **Actor:** 15 inputs -> 256 hidden -> 256 hidden -> 3 outputs (ReLU activations, Gaussian output with learnable log_std)
- **Critic:** 15 inputs -> 256 hidden -> 256 hidden -> 1 output (ReLU activations)

## Setup

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Unity 6.4 (6000.4.6f1)

### Unity Setup

1. Open the Unity project folder in Unity Hub (Unity will regenerate the Library folder automatically)
2. Attach `RocketController.cs` and `PythonBridge.cs` to the rocket GameObject
3. The rocket needs a `Rigidbody` component
4. Create an empty GameObject as the landing pad and assign it to PythonBridge's `landingPad` field
5. Set Inspector values to match `python/config.py` (see below)

### Configuration

All physics and environment constants are defined in `python/config.py`. This is the single source of truth for the Python side. Unity Inspector values must be set to match.

**config.py values:**

| Constant | Value | Notes |
|----------|-------|-------|
| DRY_MASS | 22000 | kg |
| FUEL_MASS | 2000 | kg, resets each episode |
| TWR | 2.0 | Thrust-to-weight ratio |
| MAX_THRUST | 470880 | Computed: MASS * 9.81 * TWR |
| SPAWN_HEIGHT | 200 | metres |
| PAD_HEIGHT | 5.2 | Rocket resting y-position |
| POS_SCALE | 50 | Observation scaling |
| VEL_SCALE | 20 | Observation scaling |
| ANG_VEL_SCALE | 10 | Observation scaling |

**RocketController Inspector:**

| Field | Value | Notes |
|-------|-------|-------|
| maxThrust | 470880 | Must match config.py |
| rcsForce | 2000000 | Scaled for rocket's inertia |
| dryMass | 22000 | Must match config.py |
| fuelMass | 2000 | Must match config.py |
| useAirDrag | true | Toggle aerodynamic drag |
| spawnHeight | 200 | Must match config.py |
| spawnAngleRange | 5 | Start low, increase via curriculum |

**PythonBridge Inspector:**

| Field | Value | Notes |
|-------|-------|-------|
| padHeight | 5.2 | Adjust to rocket's resting y-position |
| landingSpeedLimit | 3 | m/s, crash if exceeded at pad |
| landingTiltLimit | 10 | Degrees, crash if exceeded at pad |
| landingRadius | 15 | Metres, must land within this distance of pad centre |
| tiltCrashLimit | 45 | Mid-flight crash threshold |
| outOfBoundsHeight | 400 | Must be higher than spawnHeight |
| timeScale | 1 | Use 1 for PID/Evaluate, 10 for PPO |
| useGimbal | true | Must be true for RCS to work |
| maxEpisodeTime | 30 | Seconds (sim time) |
| posScale | 50 | Must match config.py |
| velScale | 20 | Must match config.py |
| angVelScale | 10 | Must match config.py |

## Running the Pipeline

**Important:** Always start the Python script first (it is the TCP server), then press Play in Unity (it is the client).

### Step 1: Collect PID Demonstrations

```bash
cd python
python PID_collect.py
# Then press Play in Unity (timeScale = 1)
```

Runs a suicide burn PID controller that lands the rocket and records every observation-action pair from successful landings only. Saves to `demos.npz`. Check the terminal for landing rate (should be >90%).

### Step 2: Train with Behavioral Cloning

```bash
python Behaviorcloning.py
# No Unity needed for this step
```

Trains the actor network on PID demos using MSE loss. Saves weights to `NN/actor.pth`. Check that the validation loss converges (should reach around 0.001 or lower).

### Step 3: Evaluate the BC Policy

```bash
python EvaluateUnity.py
# Then press Play in Unity (timeScale = 1)
```

Runs the trained policy deterministically (no exploration noise) for 50 episodes. Watch in Unity to confirm the rocket lands. Check the printed success rate.

### Step 4: Fine-Tune with PPO

```bash
python Unity_PPO.py
# Then press Play in Unity (timeScale = 10)
```

Loads the BC weights (`RESUME = True`) and fine-tunes with PPO. Set Unity's timeScale to 10 for faster training. Monitor the average reward and landing rate. Training log is saved to `results/training_log.json` every 50 iterations.

**PPO hyperparameters (tuned for fine-tuning, not from-scratch training):**
- Actor learning rate: 0.00003
- Critic learning rate: 0.0001
- Clip range: [0.95, 1.05] (tight, prevents large policy changes)
- Steps per iteration: 2048
- Epochs per iteration: 2
- Batch size: 64
- Entropy bonus: 0.01
- log_std reset to -2.0 on load, clamped at -2.5 minimum

### Step 5: Evaluate the PPO Policy

```bash
python EvaluateUnity.py
# Then press Play in Unity (timeScale = 1)
```

Same as Step 3. Compare landing rate and quality before and after PPO.

## Curriculum Learning

To train for harder conditions, progressively increase difficulty in the Unity Inspector and re-run PPO with `RESUME = True`:

```
Level 1: spawnAngleRange=5                       -> PID + BC + PPO
Level 2: spawnAngleRange=10                      -> PPO only (RESUME from Level 1)
Level 3: spawnAngleRange=20                      -> PPO only
Level 4: spawnAngleRange=35                      -> PPO only
Level 5: + spawnPosRange=3                       -> PPO only
Level 6: + spawnPosRange=5                       -> PPO only
Level 7: + spawnVelocityRange=2                  -> PPO only
```

Back up `NN/actor.pth` and `NN/critic.pth` before each level.

When using "PPO only", change the Inspector values in Unity and run Unity_PPO.py with `RESUME = True`. The network adapts its existing policy to harder conditions.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Unity freezes on Play | Python script not running | Start Python first, then Play |
| Rocket spirals on Y-axis | Asymmetric inertia tensor | Set rb.inertiaTensor and inertiaTensorRotation in Awake |
| Rocket spirals in PPO | log_std too high | Ensure BC sets log_std = -3.0 before saving |
| Rocket slams into ground | Shaping reward at high timeScale | Use terminal-only reward (no distance shaping) |
| Rocket hovers forever | Per-step reward too positive | Use -0.01 time penalty only |
| "Address already in use" | Port 5005 still bound | Wait 30s or kill old Python process |
| Rocket lands but counts as crash | padHeight too low | Increase padHeight to match rocket resting position |
| PID lands but too far from pad | Landing radius too small | Increase landingRadius in Inspector |
| PPO reward oscillates wildly | Learning rate too high or clip too wide | Use actor_lr=0.00003, clip [0.95, 1.05] |
| PPO stuck at -100 | BC policy not landing | Run EvaluateUnity.py first to verify BC works |
| Rocket stops thrusting mid-descent | Out of fuel | Increase fuelMass or reduce spawnHeight |
| outOfBoundsHeight crash at spawn | outOfBoundsHeight = spawnHeight | Set outOfBoundsHeight > spawnHeight (e.g. 400) |

## File Reference

| File | Purpose | When to run | Needs Unity? |
|------|---------|-------------|--------------|
| config.py | Single source of truth for constants | Imported by all scripts | No |
| PID_collect.py | Collect expert demonstrations | First | Yes (timeScale=1) |
| Behaviorcloning.py | Train NN on demonstrations | After PID | No |
| EvaluateUnity.py | Test trained policy | After BC or PPO | Yes (timeScale=1) |
| Unity_PPO.py | Fine-tune with RL | After BC | Yes (timeScale=10) |
| RocketController.cs | Rocket physics and RCS | Always in Unity | N/A |
| PythonBridge.cs | Socket bridge and rewards | Always in Unity | N/A |
