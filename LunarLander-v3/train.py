"""
SB3 PPO training for LunarLander-v3.

Target: Windows with RTX 4070 — SubprocVecEnv runs natively.

Saves:
  sb3_checkpoint.zip  — full training state for resume
  xx.zip              — best model for evaluation harness (load_parameter("xx.pt"))

Usage:
    python train.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from xxx import make_env

N_ENVS      = 8
TOTAL_STEPS = 1_000_000
SAVE_FREQ   = 50_000
EVAL_FREQ   = 25_000

RESUME_CKPT = "sb3_checkpoint"
BEST_MODEL  = "xx"


class OverwriteCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"  -> Checkpoint saved at step {self.num_timesteps:,}")
        return True


def train():
    env      = SubprocVecEnv([make_env] * N_ENVS)
    eval_env = SubprocVecEnv([make_env])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=".",
        n_eval_episodes=10,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = OverwriteCheckpointCallback(SAVE_FREQ, RESUME_CKPT, verbose=1)

    if os.path.exists(RESUME_CKPT + ".zip"):
        print(f"Resuming from {RESUME_CKPT}.zip ...")
        model     = PPO.load(RESUME_CKPT, env=env, device="auto")
        remaining = max(0, TOTAL_STEPS - model.num_timesteps)
        print(f"  Completed {model.num_timesteps:,} / {TOTAL_STEPS:,} steps — {remaining:,} remaining")
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=False,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
        )
    else:
        print("Starting fresh training...")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=1024,           # per env; 1024 × 8 = 8192 total per rollout
            batch_size=64,
            n_epochs=4,
            learning_rate=1e-3,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            normalize_advantage=True,
            verbose=1,
            tensorboard_log="./tb_logs",
        )
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
        )

    if os.path.exists("best_model.zip"):
        os.replace("best_model.zip", BEST_MODEL + ".zip")
        print(f"Best model saved as {BEST_MODEL}.zip")
    else:
        model.save(BEST_MODEL)
        print(f"Final model saved as {BEST_MODEL}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
