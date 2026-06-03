# Rocket Landing with Reinforcement Learning

A SpaceX Falcon 9-style rocket landing simulation using Proximal Policy Optimization (PPO) trained from scratch in Unity 3D. The agent learns to perform a suicide burn descent and soft vertical landing using a two-phase training pipeline: Behavioral Cloning (BC) from a PID controller, followed by PPO fine-tuning.

## Architecture

Python handles all machine learning (PyTorch). Unity handles all physics simulation. They communicate over a TCP socket on port 5005. Python is the server, Unity is the client.

```
Python (PPO / BC / PID)  <---TCP socket--->  Unity (Physics / Rendering)
         |                                          |
    Neural Network                           RocketController.cs
    Reward logic                             PythonBridge.cs
    Training loop                            Rigidbody physics
```

## Project Structure

```
├── PID_collect.py         # Step 1: PID controller collects landing demonstrations
├── Behaviorcloning.py     # Step 2: Train neural network to imitate PID
├── Unity_PPO.py           # Step 3: Fine-tune with PPO reinforcement learning
├── EvaluateUnity.py       # Step 4: Test trained policy (deterministic)
├── demos.npz              # Generated: PID demonstration data
├── NN/
│   ├── actor.pth          # Trained actor (policy) weights
│   └── critic.pth         # Trained critic (value function) weights
├── results/
│   ├── bc_training.png    # BC loss curves
│   └── training_curve.png # PPO reward curve
└── Assets/Scripts/        # Unity C# scripts
    ├── RocketController.cs
    └── PythonBridge.cs
```

## How It Works

### The Two-Phase Training Pipeline

**Phase 1 -- Behavioral Cloning (supervised learning)**

A hand-coded PID controller lands the rocket using a suicide burn algorithm. We record every (observation, action) pair as training data, then train a neural network to imitate the PID via supervised learning (MSE loss). This gives the network a competent starting policy without any reward engineering.

**Phase 2 -- PPO Fine-Tuning (reinforcement learning)**

The BC-trained network is loaded into PPO, which fine-tunes it using terminal-only rewards (+100 to +250 for landing, -100 for crashing). PPO improves landing quality (softer, more upright) and generalises to harder conditions (more tilt, position offset, initial velocity).

### Why Not Train PPO From Scratch?

We tried. Dense reward shaping (rewarding the agent for getting closer to the pad) caused two failure modes: hovering (the agent farms small per-step rewards by staying alive) and slamming (at high simulation speed, moving faster toward the pad yields more shaping reward per step). Terminal-only rewards are too sparse for a random policy to discover landing by chance. BC solves this by giving PPO a working starting point.

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

### Reward Function (terminal only)

- **Successful landing:** 100 + speed_bonus + tilt_bonus (max 250)
  - speed_bonus = (1 - speed / speed_limit) * 100
  - tilt_bonus = (1 - tilt / tilt_limit) * 50
- **Crash:** -100
- **Per step:** -0.01 (small time penalty)

No distance-based shaping. This is intentional and critical for stability at high simulation speeds.

## Setup

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Unity 2022+ with the project open

### Unity Setup

1. Open the Unity project
2. Attach `RocketController.cs` and `PythonBridge.cs` to the rocket GameObject
3. The rocket needs a `Rigidbody` component
4. Create an empty GameObject as the landing pad and assign it to PythonBridge's `landingPad` field
5. Set Inspector values (see below)

### Key Inspector Values

**RocketController:**

| Field | Value | Notes |
|-------|-------|-------|
| maxThrust | 367000 | Falcon 9 TWR 1.7 |
| rcsForce | 400000 | Scaled for 22t rocket |
| mass (Rigidbody) | 22000 | Falcon 9 landing mass |
| spawnHeight | 30 | Start with 30, increase later |
| spawnAngleRange | 5 | Degrees of random initial tilt |

**PythonBridge:**

| Field | Value | Notes |
|-------|-------|-------|
| padHeight | 3.0 | Adjust to rocket's resting y-position |
| landingSpeedLimit | 3 | m/s, crash if exceeded at pad |
| landingTiltLimit | 10 | Degrees, crash if exceeded at pad |
| tiltCrashLimit | 45 | Mid-flight crash threshold |
| timeScale | 1 | Use 1 for PID/Evaluate, 10 for PPO |
| useGimbal | true | Must be true for RCS to work |
| maxEpisodeTime | 30 | Seconds (sim time) |
| posScale | 50 | Must match Python scripts |
| velScale | 20 | Must match Python scripts |

## Running the Pipeline

**Important:** Always start the Python script first (it is the TCP server), then press Play in Unity (it is the client).

### Step 1: Collect PID Demonstrations

```bash
python PID_collect.py
# Then press Play in Unity (timeScale = 1)
```

Runs a suicide burn PID controller that lands the rocket and records every observation-action pair. Saves to `demos.npz`. Check the terminal for landing rate (should be >90%).

**Key parameters to match between PID_collect.py and Unity Inspector:**
- `MASS`, `MAX_THRUST` must match Rigidbody mass and maxThrust
- `PAD_HEIGHT` must match padHeight
- `POS_SCALE`, `VEL_SCALE`, `ANG_VEL_SCALE` must match posScale, velScale, angVelScale
- `SPAWN_HEIGHT` should match spawnHeight

### Step 2: Train with Behavioral Cloning

```bash
python Behaviorcloning.py
# No Unity needed for this step
```

Trains the actor network on PID demos using MSE loss. Saves weights to `NN/actor.pth`. Check that the validation loss converges (should reach around 0.001 or lower).

**Critical:** The script sets `log_std = -2.0` before saving. This controls exploration noise in PPO. Without this, PPO adds huge random noise to actions and the rocket spirals out of control.

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

Loads the BC weights (`RESUME = True`) and fine-tunes with PPO. Set Unity's timeScale to 10 for faster training. Monitor the average reward (should stay above 100 if BC was good).

**PPO hyperparameters (tuned for fine-tuning, not from-scratch training):**
- Learning rate: 0.0001 (low, to preserve BC policy)
- Clip range: [0.95, 1.05] (tight, prevents large policy changes)
- Steps per iteration: 512
- Epochs per iteration: 2
- Batch size: 64

### Step 5: Evaluate the PPO Policy

```bash
python EvaluateUnity.py
# Then press Play in Unity (timeScale = 1)
```

Same as Step 3. Compare landing rate and quality before and after PPO.

## Curriculum Learning

To train for harder conditions, progressively increase difficulty:

```
Level 1: spawnAngleRange=5                       -> PID + BC + PPO
Level 2: spawnAngleRange=10                      -> PPO only (RESUME from Level 1)
Level 3: spawnAngleRange=15, spawnPosRange=5     -> PPO only (if it fails, redo PID+BC)
Level 4: + spawnVelocityRange=3                  -> PPO only
Level 5: Tighten landingTiltLimit=5              -> PPO only
```

Back up `NN/actor.pth` and `NN/critic.pth` before each level.

When using "PPO only", change the Inspector values in Unity and run Unity_PPO.py with `RESUME = True`. The network adapts its existing policy to harder conditions.

When a new skill is needed (e.g. lateral correction for position offsets), collect new PID demos at that difficulty and re-run the full pipeline from Step 1.

## Technical Details

### RCS vs Gimbal

The project uses Reaction Control System (RCS) thrusters for attitude control instead of engine gimbaling. RCS applies torque directly to the rocket body (`rb.AddTorque`), independent of main engine state. This was chosen because:

- **Gimbal only works when the engine fires.** During the free-fall phase of a suicide burn, the engine is off and gimbal provides zero attitude authority. Any initial tilt grows uncorrected.
- **RCS works at all times.** The rocket can correct its orientation during free fall, enabling fuel-optimal descent profiles.
- **Simpler control mapping.** RCS torque is linear (command directly maps to torque). Gimbal torque depends on both gimbal angle AND thrust level (nonlinear coupling), which is harder for the neural network to learn.

This mirrors real rocket design. Falcon 9 uses engine gimbal, cold gas thrusters (RCS), and grid fins together. Our simulation uses RCS as the sole attitude actuator with proportionally higher authority to compensate.

### Cached Thrust (timeScale Safety)

PythonBridge applies the last received thrust command every physics frame (`FixedUpdate`), not just when a new action arrives. This is critical because at `timeScale = 10`, Unity runs approximately 10 physics frames between Python actions. Without caching, the rocket free-falls between commands and the suicide burn fails.

### Network Architecture

- **Actor:** 15 inputs -> 256 hidden -> 256 hidden -> 3 outputs (ReLU activations, Gaussian output with learnable log_std)
- **Critic:** 15 inputs -> 256 hidden -> 256 hidden -> 1 output (ReLU activations)

### Socket Protocol

Python sends 3 floats (thrust, rcsX, rcsZ). Unity replies with 17 floats (15 observations + 1 reward + 1 done flag). Reset signal: thrust = -999.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Unity freezes on Play | Python script not running | Start Python first, then Play |
| Rocket spirals in PPO | log_std too high | Ensure BC sets log_std = -2.0 before saving |
| Rocket slams into ground | Shaping reward at high timeScale | Use terminal-only reward (no distance shaping) |
| Rocket hovers forever | Per-step reward too positive | Use -0.01 time penalty only |
| "Address already in use" | Port 5005 still bound | Wait 30s or kill old Python process |
| Rocket lands but engine stays on | No engine cutoff on done | PythonBridge sets cachedThrust=0 on done |
| PID crashes from 100m | TWR 1.7 margins too tight | Increase PID margin to 10, or start from 30m |
| BC loss won't decrease | PID data has crashes mixed in | Increase PID episodes, check landing rate |

## File Reference

| File | Purpose | When to run | Needs Unity? |
|------|---------|-------------|--------------|
| PID_collect.py | Collect expert demonstrations | First | Yes (timeScale=1) |
| Behaviorcloning.py | Train NN on demonstrations | After PID | No |
| EvaluateUnity.py | Test trained policy | After BC or PPO | Yes (timeScale=1) |
| Unity_PPO.py | Fine-tune with RL | After BC | Yes (timeScale=10) |
| RocketController.cs | Rocket physics and RCS | Always in Unity | N/A |
| PythonBridge.cs | Socket bridge and rewards | Always in Unity | N/A |
