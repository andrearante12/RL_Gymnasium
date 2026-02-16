import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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

class xxxAgent:
    def __init__(self,obs_dim,act_dim,act_low,act_high,**kwargs):

    def act(self, s,**kwargs):

    def load_parameter(self,file):
