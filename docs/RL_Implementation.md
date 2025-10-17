# RL Implementation in Powergrid

This document describes how reinforcement learning (RL) is integrated into the **Powergrid** project (YanbinLin94/powergrid), how to run training/evaluation, and key design decisions.

---

## 1. Project Structure & RL Modules

The repository’s RL-related parts are organized as follows (suggested layout):

```
powergrid/
  envs/                  # existing environment modules (IEEE‑13, IEEE‑34, etc.)
  base_env.py            # base class and common logic
  rl/
    agents/               # agent / policy implementations (SAC, PPO, mixed heads, etc.)
    training/             # training scripts, evaluation scripts
    wrappers/             # observation/action normalization, reward shaping, safe wrappers
    configs/              # YAML or JSON configs for experiments
  examples/
    sb3_train_powergrid.py  # example script to train RL algorithms
docs/
  rl/
    RL_Implementation.md  # this document
    experiments.md
```

---

## 2. Dependencies & Setup

Make sure the environment has the following key dependencies:

- `pandapower` — power flow, network modeling  
- `gymnasium` — environment interface  
- `stable-baselines3` (or alternate RL library)  
- `numpy`, `pandas`, `torch` (if using PyTorch backend)  
- Others: `lxml`, `matplotlib`, etc.

Install with a command like:

```bash
pip install pandapower gymnasium stable-baselines3 torch lxml matplotlib pandas
```

Ensure your virtual environment is active when running RL scripts.

---

## 3. Environment Interface & Wrappers

The RL integration uses the existing powergrid environments, with wrappers to adapt to RL requirements:

- **Environment creation**:  
  You can instantiate environments like `IEEE13Env` or `IEEE34Env` with parameters, e.g. `episode_length`, `train` mode, etc.

- **Action space normalization**:  
  A wrapper `NormalizeActionWrapper` is used so the RL agent acts in `[-1, 1]`, while the environment rescales those actions to physical ranges (e.g. real power, reactive power, tap steps).

- **Observation normalization (optional)**:  
  A wrapper such as `ObsRunningNorm` can be used to normalize observations (mean/std) online to help stabilize learning.

- **Reward shaping**:  
  Custom wrappers or reward functions can combine economic costs and safety penalties (voltage violations, line loading, SOC limits, tap wear, etc.).

- **Safety checks / termination logic**:  
  The environment should gracefully terminate an episode if safety constraints are breached or if power flow fails to converge. The RL code must anticipate episodes truncating or terminating early.

---

## 4. Agent / Policy Architecture

Typical RL algorithms used are **SAC**, **PPO**, **TD3**, or **DDPG** depending on whether the action space is fully continuous or mixed (continuous + discrete taps). Some design decisions:

- Use multi-layer MLPs (e.g. 2 hidden layers of 256 units) as policy and value networks rather than very deep networks, to reduce training cost.
- For **mixed action spaces**, PPO can support multiple output heads (one for continuous setpoints, one for discrete taps). You may need to customize the policy class accordingly.
- The agent training code should support vectorized environments (via `DummyVecEnv` or `SubprocVecEnv`) for throughput.

---

## 5. Training & Evaluation Workflow

### Training

1. Parse command-line arguments: environment name, algorithm, total timesteps, logging directory, seed, normalization flags, etc.
2. Create vectorized training environment(s) and optional wrappers.
3. Optionally create an evaluation environment with deterministic/noise off.
4. Instantiate RL model (e.g. `SAC(policy, env, **hyperparams)`).
5. Set up callbacks:
   - **CheckpointCallback**: periodically save model snapshots.
   - **EvalCallback**: periodically run evaluation episodes, log metrics, and save best models.
6. Run `.learn(total_timesteps, callback=…)`.
7. Save final model after training.

### Evaluation

- Use deterministic policy (no exploration noise).
- Run multiple episodes in evaluation env.
- Compute average and standard deviation of returns, count safety violations, log other metrics.
- You may need to unnormalize or re-scale observations/actions if wrappers were used.

---

## 6. Reward Design & Scaling

The reward function is crucial. A typical reward formula:

```
reward = - (generation_cost + curtailment_cost + settlement_cost)
         - λ_V * voltage_violation_penalty
         - λ_L * line_loading_penalty
         - λ_SOC * soc_violation
         - λ_tap * tap_move_penalty
         - λ_act * ∥Δaction∥²
```

- Use **dense rewards** to give feedback every timestep.
- Scale each penalty term relative to historical magnitudes (e.g. use 95th percentile) so that no single term dominates.
- Use **curriculum** training: start with light penalties, then gradually increase λ weights as the model improves.

---

## 7. Common Pitfalls & Debug Tips

- **Index out of bounds / dataset lengths**: make sure episode lengths never exceed the dataset’s capacity, or wrap indices safely.
- **Power flow non-convergence**: some action combinations may cause pandapower to fail. In such cases, terminate episode with large negative reward and restart.
- **Mixed action instability**: discrete and continuous heads may need balanced learning rates. Temperature annealing or constraint losses may help.
- **Reward scale mismatch**: if rewards are too large/small, normalize them or clip them to avoid exploding gradients.
- **Overfitting to training days**: use domain randomization (vary loads, PV profiles) and hold out test days.

---

## 8. Example: Minimal Training Script

```python
from powergrid.envs.single_agent.ieee13_mg import IEEE13Env
from stable_baselines3 import SAC

env = IEEE13Env({"episode_length": 24, "train": True})
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("sac_ieee13_final")
```

---

## 9. Suggested Experiments & Ablations

- Compare SAC vs PPO vs TD3 on IEEE‑13 and IEEE‑34.
- Ablate reward terms: remove voltage penalty, tap penalty, action smoothing penalty.
- Vary episode length (24, 48, 96 steps) and test generalization.
- Domain randomization: perturb loads, PV, grid parameters.
- Multi-agent variant (if implemented): compare centralized vs decentralized control.
