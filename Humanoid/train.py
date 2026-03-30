"""
SB3 PPO training for Humanoid-v5 with VecNormalize.

Target: Windows with RTX 4070 — SubprocVecEnv runs natively.

Saves:
  sb3_checkpoint.zip  — model weights + optimizer state for resume
  vecnorm.pkl         — observation normalization statistics (updated at checkpoints)
  xxx.zip             — best model for evaluation harness (load_parameter("xxx.pt"))

Usage:
    python train.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from xxx import make_env

N_ENVS      = 8
TOTAL_STEPS = 20_000_000
SAVE_FREQ   = 200_000
EVAL_FREQ   = 100_000

RESUME_CKPT = "sb3_checkpoint"
VECNORM_PKL = "vecnorm.pkl"
BEST_MODEL  = "xxx"


class CheckpointWithVecNormCallback(BaseCallback):
    """Saves model + vecnorm stats together so they stay in sync."""

    def __init__(self, save_freq, model_path, vecnorm_path, verbose=1):
        super().__init__(verbose)
        self.save_freq    = save_freq
        self.model_path   = model_path
        self.vecnorm_path = vecnorm_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            self.model.save(self.model_path)
            self.training_env.save(self.vecnorm_path)
            if self.verbose:
                print(f"  -> Checkpoint + vecnorm saved at step {self.num_timesteps:,}")
        return True


def train():
    raw_env  = SubprocVecEnv([make_env] * N_ENVS)
    raw_eval = SubprocVecEnv([make_env])

    if os.path.exists(RESUME_CKPT + ".zip") and os.path.exists(VECNORM_PKL):
        print(f"Resuming from {RESUME_CKPT}.zip ...")
        env      = VecNormalize.load(VECNORM_PKL, raw_env)
        eval_env = VecNormalize.load(VECNORM_PKL, raw_eval)
        env.training          = True
        env.norm_reward       = True
        eval_env.training     = False
        eval_env.norm_reward  = False
        model = PPO.load(RESUME_CKPT, env=env, device="auto",
            learning_rate=1e-4,
            target_kl=0.02,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        remaining = TOTAL_STEPS
        print(f"  Resuming with stability fixes — {remaining:,} steps")
    else:
        print("Starting fresh training...")
        env      = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(raw_eval, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=".",
        n_eval_episodes=5,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointWithVecNormCallback(SAVE_FREQ, RESUME_CKPT, VECNORM_PKL, verbose=1)

    if os.path.exists(RESUME_CKPT + ".zip") and os.path.exists(VECNORM_PKL):
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=True,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,            # per env; 512 × 4 = 2048 total per rollout
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log="./tb_logs",
        )
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
        )

    # Save final vecnorm stats (always up-to-date after training)
    env.save(VECNORM_PKL)

    if os.path.exists("best_model.zip"):
        os.replace("best_model.zip", BEST_MODEL + ".zip")
        print(f"Best model saved as {BEST_MODEL}.zip")
    else:
        model.save(BEST_MODEL)
        print(f"Final model saved as {BEST_MODEL}.zip")

    print(f"VecNormalize stats saved to {VECNORM_PKL}")
    env.close()
    raw_eval.close()


if __name__ == "__main__":
    train()
