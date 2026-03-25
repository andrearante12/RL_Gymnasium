# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE6882 Reinforcement Learning Project 2 — implement RL agents for three Gymnasium environments:
- **CarRacing-v3** — image-based continuous control (Box2D)
- **LunarLander-v3** — vector-state discrete control (Box2D)
- **Humanoid-v5** — high-dimensional continuous control (MuJoCo)

Agents are evaluated on 8 testcases (2 visible sample + 6 hidden). Final score is aggregated episode returns.

## Running Evaluations

Each environment has its own evaluation script. Run from within each subdirectory:

```bash
# CarRacing (note: typo in filename — evaulation.py, not evaluation.py)
cd CarRacing-v3 && python evaulation.py

# LunarLander
cd LunarLander-v3 && python evaluation.py

# Humanoid
cd Humanoid && python evaluation.py
```

Each script runs 2 sample testcases (seeds 0 and 2), records GIF videos, and prints per-testcase scores and aggregate total.

## Architecture

### Pattern per environment

Each environment folder contains:
- `xxx.py` — **implement here**: `make_env()` factory and `xxxAgent` class
- `evaluation.py` — evaluation harness (do not modify)
- `sample_testcase/` — reference GIF videos

### Agent class contract

Every agent must implement exactly:
```python
def __init__(self, obs_dim, act_dim, ...):  # initialize model
def act(self, state):                        # return action (and optionally log_prob, value)
def load_parameter(self, file):              # load weights from checkpoint (.pt file)
```

The `act()` signature must not change — the evaluation harness calls it directly.

### Environment-specific details

**CarRacing-v3**
- Input: RGB image frames; needs CNN-based policy (PPO recommended)
- Output: continuous action
- Agent inherits from `nn.Module`
- Checkpoint file: `xxx.pt`

**LunarLander-v3**
- Input: 8-dim vector state; output: discrete action (4 choices)
- Fixed env params (cannot change): `gravity=-10.0`, `enable_wind=True`, `wind_power=15.0`, `turbulence_power=1.5`
- Agent is plain Python class (not `nn.Module`)
- Checkpoint file: `xxx.pt`

**Humanoid-v5**
- Input: high-dimensional continuous state; output: continuous action
- Constructor signature: `__init__(obs_dim, act_dim, act_low, act_high, **kwargs)`
- `act()` must return tuple: `(action, log_prob, value)`
- Wrapped with `AwkwardStartWrapper` — randomizes initial pose; testcase 1 uses 25% awkward probability, testcase 2 uses 70%
- Checkpoint file: `xx.pt`
- Rendering: EGL (headless MuJoCo)

## Submission

Rename `xxx.py` to `{groupname}.py`, then submit a folder `{groupname}_project2/` containing:
- `{groupname}.py`
- `evaluation.py`

Submit to Google Drive (link in README.md).
