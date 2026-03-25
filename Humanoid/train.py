"""
PPO training script for Humanoid-v5.

Usage:
    python train.py

Two files are maintained:
  checkpoint.pt  — full training state (model + optimizer + counters), for resume
  xxx.pt         — best model weights only, for evaluation

Interrupt with Ctrl-C at any time; restart with the same command to resume.
Note: Humanoid is hard — expect meaningful locomotion after ~5-10M steps.
"""

import os
import numpy as np
import torch
import torch.optim as optim

from xxx import xxxAgent, make_env

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_STEPS       = 2048
N_EPOCHS      = 10
BATCH_SIZE    = 64
LR            = 3e-4
GAMMA         = 0.99
LAMBDA        = 0.95
CLIP_EPS      = 0.2
VF_COEF       = 0.5
ENT_COEF      = 0.005
MAX_GRAD_NORM = 0.5
TOTAL_STEPS   = 10_000_000
SAVE_INTERVAL = 200_000
LOG_INTERVAL  = 10

BEST_CKPT   = "xxx.pt"
RESUME_CKPT = "checkpoint.pt"


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    env = make_env()
    obs_dim  = int(np.prod(env.observation_space.shape))
    act_dim  = int(np.prod(env.action_space.shape))
    act_low  = env.action_space.low
    act_high = env.action_space.high

    agent = xxxAgent(obs_dim, act_dim, act_low, act_high)
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    device = agent.device

    total_steps      = 0
    best_mean_return = -float('inf')
    episode_returns  = []
    last_save        = 0

    if os.path.exists(RESUME_CKPT):
        state = torch.load(RESUME_CKPT, map_location=device, weights_only=True)
        agent.policy.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        total_steps      = state['total_steps']
        best_mean_return = state['best_mean_return']
        episode_returns  = state['episode_returns']
        last_save        = total_steps
        print(f"Resumed from step {total_steps:,} | best return so far: {best_mean_return:.1f}")
    else:
        print(f"Starting fresh training on: {device}")

    print(f"Target: {TOTAL_STEPS:,} steps")

    obs, _ = env.reset()
    rollout_count  = 0
    current_return = 0.0

    while total_steps < TOTAL_STEPS:
        # ------------------------------------------------------------------
        # 1. Collect rollout
        # ------------------------------------------------------------------
        obs_buf  = np.zeros((N_STEPS, obs_dim), dtype=np.float32)
        raw_buf  = np.zeros((N_STEPS, act_dim), dtype=np.float32)
        rew_buf  = np.zeros(N_STEPS, dtype=np.float32)
        val_buf  = np.zeros(N_STEPS, dtype=np.float32)
        lp_buf   = np.zeros(N_STEPS, dtype=np.float32)
        done_buf = np.zeros(N_STEPS, dtype=np.float32)

        for step in range(N_STEPS):
            s_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                env_action, raw_sample, log_prob, value, _ = agent.forward_train(s_tensor)

            env_action_np = env_action.squeeze(0).cpu().numpy()
            raw_np        = raw_sample.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(env_action_np)
            done = terminated or truncated
            current_return += reward

            obs_buf[step]  = obs
            raw_buf[step]  = raw_np
            rew_buf[step]  = reward
            val_buf[step]  = value
            lp_buf[step]   = log_prob
            done_buf[step] = float(done)

            obs = next_obs
            total_steps += 1

            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                obs, _ = env.reset()

        # ------------------------------------------------------------------
        # 2. Bootstrap value
        # ------------------------------------------------------------------
        with torch.no_grad():
            s_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, next_value, _ = agent.forward_train(s_tensor)
            next_value = float(next_value)

        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ------------------------------------------------------------------
        # 3. PPO update
        # ------------------------------------------------------------------
        obs_t    = torch.as_tensor(obs_buf,    dtype=torch.float32, device=device)
        raw_t    = torch.as_tensor(raw_buf,    dtype=torch.float32, device=device)
        old_lp_t = torch.as_tensor(lp_buf,     dtype=torch.float32, device=device)
        adv_t    = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret_t    = torch.as_tensor(returns,    dtype=torch.float32, device=device)

        indices = np.arange(N_STEPS)
        for _ in range(N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, N_STEPS, BATCH_SIZE):
                batch_idx = indices[start:start + BATCH_SIZE]
                log_probs, values, entropy = agent.evaluate_actions(obs_t[batch_idx], raw_t[batch_idx])
                ratio     = torch.exp(log_probs - old_lp_t[batch_idx])
                adv_batch = adv_t[batch_idx]
                pg_loss   = torch.max(
                    -adv_batch * ratio,
                    -adv_batch * ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS),
                ).mean()
                vf_loss  = ((values - ret_t[batch_idx]) ** 2).mean()
                ent_loss = -entropy.mean()
                loss = pg_loss + VF_COEF * vf_loss + ENT_COEF * ent_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        rollout_count += 1

        # ------------------------------------------------------------------
        # 4. Best model tracking
        # ------------------------------------------------------------------
        if episode_returns:
            recent_mean = float(np.mean(episode_returns[-10:]))
            if recent_mean > best_mean_return:
                best_mean_return = recent_mean
                torch.save(agent.policy.state_dict(), BEST_CKPT)
                print(f"  -> New best ({recent_mean:.1f}), saved to {BEST_CKPT}")

        # ------------------------------------------------------------------
        # 5. Logging & resume checkpoint
        # ------------------------------------------------------------------
        if rollout_count % LOG_INTERVAL == 0 and episode_returns:
            recent = episode_returns[-10:]
            print(
                f"Steps: {total_steps:>10,} | "
                f"Episodes: {len(episode_returns):>5} | "
                f"Mean (last 10): {np.mean(recent):>8.1f} | "
                f"Best: {best_mean_return:>8.1f}"
            )

        if total_steps - last_save >= SAVE_INTERVAL:
            torch.save({
                'model':            agent.policy.state_dict(),
                'optimizer':        optimizer.state_dict(),
                'total_steps':      total_steps,
                'best_mean_return': best_mean_return,
                'episode_returns':  episode_returns,
            }, RESUME_CKPT)
            last_save = total_steps
            print(f"  -> Resume checkpoint saved at step {total_steps:,}")

    torch.save({
        'model':            agent.policy.state_dict(),
        'optimizer':        optimizer.state_dict(),
        'total_steps':      total_steps,
        'best_mean_return': best_mean_return,
        'episode_returns':  episode_returns,
    }, RESUME_CKPT)
    print(f"Training complete. Best return: {best_mean_return:.1f} (saved to {BEST_CKPT})")
    env.close()


if __name__ == "__main__":
    train()
