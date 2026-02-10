from tqdm import tqdm
import os

# Pick ONE depending on your machine:
os.environ.setdefault("MUJOCO_GL", "egl") 
import numpy as np
from Humanoidagent import HumanoidAgent
import torch
import gymnasium as gym

def make_env(render_mode=None, testcase=None):
    env = gym.make("Humanoid-v5", render_mode=render_mode)

    # testcase can be None / dict
    if testcase is not None:
        env = AwkwardStartWrapper(env, **testcase)

    return env


class AwkwardStartWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        awkward_prob: float = 0.5,
        z_drop_range=(0.05, 0.20),
        quat_noise=0.08,
        joint_noise=0.25,
        vel_noise=0.60,
        min_z: float = 1.0,
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
            z_drop = rng.uniform(self.z_drop_range[0], self.z_drop_range[1])
            qpos[2] = max(qpos[2] - z_drop, self.min_z)
            quat = qpos[3:7].copy()
            quat += rng.uniform(-self.quat_noise, self.quat_noise, size=quat.shape)
            quat /= (np.linalg.norm(quat) + 1e-8)
            qpos[3:7] = quat
            
            qpos[7:] += rng.uniform(-self.joint_noise, self.joint_noise, size=qpos[7:].shape)
           
            qvel += rng.uniform(-self.vel_noise, self.vel_noise, size=qvel.shape)
            u.set_state(qpos, qvel)
            obs = u._get_obs()
        return obs, info


import imageio.v2 as imageio

def evaluate(
    ckpt_path: str,
    testcase,
    testcaseid,
    seeds=(0, 2),
    max_steps: int = 1200,
    render_gif: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Todo: define your own agent
    agent = HumanoidAgent()
    agent.load(ckpt_path)

    results = []
    
    env = make_env(render_mode="rgb_array",testcase=testcase)
    obs, info = env.reset(seed=seeds)
    frames = []
    ep_ret = 0.0

    while True:
            if render_gif:
                frames.append(env.render())

            #Todo, define your agents' action function
            a, _, _ = agent.act(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)

            if terminated or truncated:
                break

    env.close()

    if render_gif:
        imageio.mimsave(f"humanoid_seed_{seeds}.gif", frames, fps=100)

    print(f"Testcase {testcaseid}: score={ep_ret:.2f}")
    results.append(ep_ret)
    return results

if __name__ == "__main__":

    #Todo: define your saved agent parameter if you have
    ckpt = "humanoid.pt"

    testcase = {
        "first": dict(
            awkward_prob=0.25,
            z_drop_range=(0.03, 0.10),
            quat_noise=0.04,
            joint_noise=0.12,
            vel_noise=0.25,
            min_z=1.05,
        ),
        "second": dict(
            awkward_prob=0.70,
            z_drop_range=(0.08, 0.22),
            quat_noise=0.10,
            joint_noise=0.30,
            vel_noise=0.70,
            min_z=1.00,
        ),
    }

    # Evaluate two testcase
    _ = evaluate(
        ckpt_path=ckpt,
        testcase=testcase['first'],
        testcaseid=1,
        seeds=(0),
        max_steps=1200,
        render_gif=True,
        device="cuda"
    )
    _ = evaluate(
        ckpt_path=ckpt,
        testcase=testcase['second'],
        testcaseid=2,
        seeds=(10),
        max_steps=1200,
        render_gif=True,
        device="cuda"
    )
