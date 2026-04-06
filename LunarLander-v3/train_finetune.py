"""Finetune LunarLander-v3 from best SB3 checkpoint with domain randomization."""
import os
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from xxx import make_env

N_ENVS      = 8
TOTAL_STEPS = 5_000_000
SAVE_FREQ   = 100_000
EVAL_FREQ   = 25_000

SEED_MODEL  = "xx"          # best model from original run (eval 229)
RESUME_CKPT = "sb3_ft_ckpt"
BEST_MODEL  = "xx"


def make_env_dr(render_mode=None):
    """Domain-randomized env for robust training."""
    return make_env(
        render_mode=render_mode,
        enable_wind=True,
        gravity=random.uniform(-11.0, -9.0),
        wind_power=random.uniform(10.0, 20.0),
        turbulence_power=random.uniform(1.0, 2.0),
    )


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


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def train():
    env      = SubprocVecEnv([make_env_dr] * N_ENVS)
    eval_env = SubprocVecEnv([lambda: make_env(enable_wind=True)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=".",
        n_eval_episodes=20,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = OverwriteCheckpointCallback(SAVE_FREQ, RESUME_CKPT, verbose=1)

    if os.path.exists(RESUME_CKPT + ".zip"):
        print(f"Resuming from {RESUME_CKPT}.zip ...")
        model = PPO.load(RESUME_CKPT, env=env, device="auto",
                         learning_rate=linear_schedule(1e-4),
                         target_kl=0.015)
        remaining = max(0, TOTAL_STEPS - model.num_timesteps)
        print(f"  {model.num_timesteps:,} done — {remaining:,} remaining")
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=False,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
        )
    else:
        print(f"Finetuning from {SEED_MODEL}.zip ...")
        model = PPO.load(SEED_MODEL, env=env, device="auto",
                         learning_rate=linear_schedule(1e-4),
                         n_steps=1024,
                         batch_size=64,
                         n_epochs=4,
                         gamma=0.999,
                         gae_lambda=0.98,
                         ent_coef=0.005,
                         target_kl=0.015,
                         tensorboard_log="./tb_logs")
        model.learn(
            total_timesteps=TOTAL_STEPS,
            reset_num_timesteps=True,
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
