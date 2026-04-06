import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from xxx import make_env, DiscreteActionWrapper

N_ENVS      = 8
TOTAL_STEPS = 10_000_000
SAVE_FREQ   = 100_000       # total env steps between resume checkpoints
EVAL_FREQ   = 50_000        # total env steps between evaluations

RESUME_CKPT = "sb3_checkpoint"
VECNORM_PKL = "vecnorm.pkl"
BEST_MODEL  = "xxx"


# Checkpoint callback — overwrites a single file so resuming is simple

class CheckpointWithVecNormCallback(BaseCallback):
    def __init__(self, save_freq, model_path, vecnorm_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            self.model.save(self.model_path)
            self.training_env.save(self.vecnorm_path)
            if self.verbose:
                print(f"  -> Checkpoint + vecnorm saved at step {self.num_timesteps:,}")
        return True


# Training

def make_env_discrete(render_mode=None, training=False):
    """Wrap with DiscreteActionWrapper on top of the reward-shaped env."""
    env = make_env(render_mode=render_mode, training=training)
    env = DiscreteActionWrapper(env)
    return env


def train():
    raw_env  = SubprocVecEnv([lambda: make_env_discrete(training=True)] * N_ENVS)
    raw_eval = SubprocVecEnv([lambda: make_env_discrete(training=False)])

    if os.path.exists(RESUME_CKPT + ".zip") and os.path.exists(VECNORM_PKL):
        print(f"Resuming from {RESUME_CKPT}.zip ...")
        env      = VecNormalize.load(VECNORM_PKL, raw_env)
        eval_env = VecNormalize.load(VECNORM_PKL, raw_eval)
        env.training          = True
        env.norm_reward       = True
        eval_env.training     = False
        eval_env.norm_reward  = False
    else:
        env      = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_env = VecNormalize(raw_eval, norm_obs=False, norm_reward=False)
        eval_env.training = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=".",   # saves ./best_model.zip
        n_eval_episodes=10,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointWithVecNormCallback(SAVE_FREQ, RESUME_CKPT, VECNORM_PKL, verbose=1)

    hparams = dict(
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        learning_rate=1e-4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
    )

    if os.path.exists(RESUME_CKPT + ".zip") and os.path.exists(VECNORM_PKL):
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

    env.save(VECNORM_PKL)

    if os.path.exists("best_model.zip"):
        os.replace("best_model.zip", BEST_MODEL + ".zip")
        print(f"Best model saved as {BEST_MODEL}.zip")
    else:
        model.save(BEST_MODEL)
        print(f"Final model saved as {BEST_MODEL}.zip (no eval best found)")

    print(f"VecNormalize stats saved to {VECNORM_PKL}")
    env.close()
    raw_eval.close()


if __name__ == "__main__":
    train()
