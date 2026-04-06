# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE6882 Reinforcement Learning course project. Three Gymnasium (gym) environments solved with PPO agents using PyTorch and Stable Baselines3:

- **LunarLander-v3** — Discrete actions, MLP policy, custom PyTorch PPO agent
- **CarRacing-v3** — Continuous actions (Beta distribution), CNN policy (Nature DQN backbone), custom wrappers for grayscale/crop/frame-stack
- **Humanoid-v5** — Continuous actions (Gaussian), MLP policy, uses VecNormalize for observation/reward normalization and an AwkwardStartWrapper for domain randomization

## Key Commands

```bash
# Train an agent (from within its directory)
python train.py

# Evaluate an agent (runs 2 sample testcases, produces GIF outputs)
python evaluation.py
```

Training supports resumption from `sb3_checkpoint.zip`. Humanoid also requires `vecnorm.pkl` for VecNormalize state.

## Architecture

Each environment directory follows the same structure:

| File | Purpose |
|------|---------|
| `Arante_Andre.py` | Agent class + `make_env()` — the submission file (renamed from `xxx.py`) |
| `train.py` | Training script using SB3's PPO with checkpoint/eval callbacks |
| `evaluation.py` | Evaluation harness — imports from `Arante_Andre.py`, runs testcases, saves GIFs |

**Agent interface contract** (required by evaluation harness):
- `__init__` — accepts env dimensions
- `act(state)` — returns action given observation
- `load_parameter(file)` — loads weights; supports both `.pt` (raw state_dict) and `.zip` (SB3 checkpoint)

**CarRacing-v3 specifics:** `utils.py` contains the `ActorCritic` CNN module. `make_env(training=True)` adds `NegativeRewardTerminator` and `DiscreteActionWrapper` (used only during training; evaluation uses continuous actions via Beta distribution).

**Humanoid specifics:** `make_env` in `Arante_Andre.py` optionally wraps with `AwkwardStartWrapper`. Training uses `VecNormalize` — the agent's `load_parameter` automatically loads `vecnorm.pkl` from the same directory for inference-time normalization.

## Dependencies

- `gymnasium` (with Box2D and MuJoCo)
- `stable-baselines3`
- `torch`
- `opencv-python` (CarRacing preprocessing)
- `imageio` (GIF recording in evaluation)
