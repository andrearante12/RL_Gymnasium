"""
SB3 PPO training for CarRacing-v3.

Target: Windows with RTX 4070 — SubprocVecEnv runs natively.

Saves:
  sb3_checkpoint.zip  — full training state for resume (overwritten each interval)
  xxx.zip             — best model for evaluation harness (load_parameter("xxx.pt"))

Usage:
    python train.py          # start or resume automatically
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from xxx import make_env

N_ENVS      = 8
TOTAL_STEPS = 10_000_000
SAVE_FREQ   = 100_000       # total env steps between resume checkpoints
EVAL_FREQ   = 50_000        # total env steps between evaluations

RESUME_CKPT = "sb3_checkpoint"
BEST_MODEL  = "xxx"


# ---------------------------------------------------------------------------
# Checkpoint callback — overwrites a single file so resuming is simple
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    env      = SubprocVecEnv([lambda: make_env(training=True)] * N_ENVS)
    eval_env = SubprocVecEnv([lambda: make_env(training=True)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=".",   # saves ./best_model.zip
        n_eval_episodes=10,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = OverwriteCheckpointCallback(SAVE_FREQ, RESUME_CKPT, verbose=1)

    hparams = dict(
        n_steps=512,            # balance between throughput and update frequency
        batch_size=256,
        n_epochs=4,             # fewer epochs to spend more time collecting
        learning_rate=3e-4,     # SB3 default
        clip_range=0.2,         # SB3 default
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

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
            "CnnPolicy",
            env,
            device="auto",
            policy_kwargs=dict(normalize_images=False),
            **hparams,
            verbose=1,
            tensorboard_log="./tb_logs",
        )
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
        )

    # Rename EvalCallback's best model to the filename the eval harness expects
    if os.path.exists("best_model.zip"):
        os.replace("best_model.zip", BEST_MODEL + ".zip")
        print(f"Best model saved as {BEST_MODEL}.zip")
    else:
        model.save(BEST_MODEL)
        print(f"Final model saved as {BEST_MODEL}.zip (no eval best found)")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
