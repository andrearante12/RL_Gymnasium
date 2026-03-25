import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class AwkwardStartWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        awkward_prob,
        z_drop_range,
        quat_noise,
        joint_noise,
        vel_noise,
        min_z,
    ):
        super().__init__(env)
        self.awkward_prob = awkward_prob
        self.z_drop_range = z_drop_range
        self.quat_noise = quat_noise
        self.joint_noise = joint_noise
        self.vel_noise = vel_noise
        self.min_z = min_z

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped  # HumanoidEnv
        rng = getattr(u, "np_random", None)
        if rng is None:
            # fallback (rare)
            rng = np.random.default_rng()

        if rng.random() < self.awkward_prob:
            qpos = u.data.qpos.copy()
            qvel = u.data.qvel.copy()
            # 1) crouch a bit
            z_drop = rng.uniform(self.z_drop_range[0], self.z_drop_range[1])
            qpos[2] = max(qpos[2] - z_drop, self.min_z)
            # 2) tilt/perturb torso orientation (root quaternion)
            quat = qpos[3:7].copy()
            quat += rng.uniform(-self.quat_noise, self.quat_noise, size=quat.shape)
            quat /= (np.linalg.norm(quat) + 1e-8)
            qpos[3:7] = quat
            # 3) joint angle awkward pose
            qpos[7:] += rng.uniform(-self.joint_noise, self.joint_noise, size=qpos[7:].shape)
            # 4) initial velocity noise
            qvel += rng.uniform(-self.vel_noise, self.vel_noise, size=qvel.shape)
            u.set_state(qpos, qvel)
            obs = u._get_obs()
        return obs, info


def make_env(render_mode=None, testcase=None):
    env = gym.make("Humanoid-v5", render_mode=render_mode)

    # testcase can be None / dict
    if testcase is not None:
        env = AwkwardStartWrapper(env, **testcase)
    return env


# ---------------------------------------------------------------------------
# Policy network (actor + critic with shared MLP body)
# ---------------------------------------------------------------------------

class _Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.log_std    = nn.Parameter(torch.zeros(act_dim))   # learnable, shared across states
        self.critic     = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        mean  = self.actor_mean(h)
        std   = torch.exp(self.log_std.clamp(-5, 2))
        value = self.critic(h).squeeze(-1)
        return mean, std, value


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class xxxAgent:
    """
    PPO agent for Humanoid-v5 with Gaussian (Normal) continuous actions.

    act(s, deterministic=True) → (action, log_prob, value)
    """

    def __init__(self, obs_dim, act_dim, act_low, act_high, **kwargs):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act_low  = torch.as_tensor(act_low,  dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self.policy  = _Policy(obs_dim, act_dim).to(self.device)
        self.model    = None   # set when loaded from SB3 checkpoint
        self.norm_env = None   # VecNormalize wrapper for obs normalization at inference

    def parameters(self):
        return self.policy.parameters()

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def _scale_action(self, raw):
        """Clamp raw Gaussian sample to [act_low, act_high]."""
        return raw.clamp(self.act_low, self.act_high)

    def act(self, s, deterministic=False, **kwargs):
        """
        Returns (action_numpy, log_prob_scalar, value_scalar).
        Evaluation harness unpacks as: a, _, _ = agent.act(obs, deterministic=True)
        """
        if self.model is not None:
            # SB3 path: normalize obs if VecNormalize stats were loaded
            if self.norm_env is not None:
                import numpy as np
                obs_norm = self.norm_env.normalize_obs(np.array([s]))[0]
            else:
                obs_norm = s
            action, _ = self.model.predict(obs_norm, deterministic=deterministic)
            return action, 0.0, 0.0
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, std, value = self.policy(x)
            dist  = torch.distributions.Normal(mean, std)
            if deterministic:
                sample = mean
            else:
                sample = dist.sample()
            log_prob = dist.log_prob(sample).sum(dim=-1)
            action   = self._scale_action(sample).squeeze(0).cpu().numpy()
        return action, log_prob.item(), value.item()

    def forward_train(self, s_tensor):
        """
        Training forward.
        Returns (env_action, raw_sample, log_prob, value, entropy).
          env_action  — clamped to [act_low, act_high], pass to env.step()
          raw_sample  — unclamped Normal sample, store in buffer for log_prob recomputation
        """
        mean, std, value = self.policy(s_tensor)
        dist       = torch.distributions.Normal(mean, std)
        raw_sample = dist.sample()
        log_prob   = dist.log_prob(raw_sample).sum(dim=-1)
        entropy    = dist.entropy().sum(dim=-1)
        env_action = self._scale_action(raw_sample)
        return env_action, raw_sample, log_prob, value, entropy

    def evaluate_actions(self, s_tensor, raw_samples_tensor):
        """Recompute log_probs/values for stored raw (pre-clamp) Normal samples."""
        mean, std, value = self.policy(s_tensor)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_samples_tensor).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    def load(self, file, **kwargs):
        import os
        base     = os.path.splitext(file)[0]    # "xxx.pt" → "xxx"
        zip_path = base + ".zip"
        if os.path.exists(zip_path):
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            self.model = PPO.load(base, device=self.device)
            # Load normalization stats if they exist alongside the model
            vecnorm_path = os.path.join(os.path.dirname(base) or ".", "vecnorm.pkl")
            if os.path.exists(vecnorm_path):
                venv = DummyVecEnv([make_env])
                self.norm_env = VecNormalize.load(vecnorm_path, venv)
                self.norm_env.training    = False
                self.norm_env.norm_reward = False
        else:
            state = torch.load(file, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(state)
        self.policy.eval()

    # alias so both names work
    def load_parameter(self, file, **kwargs):
        self.load(file, **kwargs)

    def save(self, file):
        torch.save(self.policy.state_dict(), file)
