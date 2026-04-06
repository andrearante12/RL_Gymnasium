from tqdm import tqdm
import os

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from Arante_Andre import xxxAgent,make_env
import torch
import gymnasium as gym
import pdb
import imageio.v2 as imageio


def evaluate(agent,testcase):
    # build env to get dims/bounds
    results=[]
    index = 0
    for key in testcase:
        frames = []
        env = make_env(render_mode="rgb_array",testcase=testcase[key])
        obs, _ = env.reset(seed=0)
        #pdb.set_trace()
        ep_ret = 0.0

        while True:
                frames.append(env.render())

                a, _, _ = agent.act(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(a)
                ep_ret += float(r)

                if terminated or truncated:
                    break
        env.close()
        imageio.mimsave(f"humanoid_{index}.gif", frames, fps=100)
        print(f"Testcase {index}: score={ep_ret:.2f}")
        results.append(ep_ret)
        index+=1
    return sum(results)

if __name__ == "__main__":
    #agent parameter
    ckpt = "xxx.pt"
    #two test cases for reference 
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
    #initialize the agent
    tmp = make_env(render_mode="rgb_array")
    obs_dim = int(np.prod(tmp.observation_space.shape))
    act_dim = int(np.prod(tmp.action_space.shape))
    act_low, act_high = tmp.action_space.low, tmp.action_space.high
    agent = xxxAgent(obs_dim, act_dim, act_low, act_high)
    #load the parameter
    agent.load(ckpt)

    # Evaluate
    scores = evaluate(agent,testcase)

    print("eventual score is: ", scores)
