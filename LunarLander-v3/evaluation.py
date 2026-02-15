from tqdm import tqdm
import gymnasium as gym
import torch
import numpy as np
import imageio.v2 as imageio
from xxxAgent import make_env,xxxAgent

def evaluate(
    agent,
    env,
    testcase=[0, 2],
):
    results = []

    for i, seed in enumerate(testcase, start=1):
        frames=[]
        s, _ = env.reset(seed=int(seed))
        ep_ret = 0.0
        ep_len = 0
        while True:
            frames.append(env.render())
            a = agent.act(s, greedy=True)
            s, r, terminated, truncated, info = env.step(a)
            ep_ret += float(r)
            ep_len += 1
            if terminated or truncated:
                break
        env.close()
        imageio.mimsave("testcase_{}.gif".format(seed), frames, fps=30)
        adjusted = ep_ret #- 0.01 * (angle_sum + angvel_sum)
        print(f"[Eval {i}] seed={seed:>3} | score={adjusted:8.2f}")
        results.append(ep_ret)

    return results


if __name__ == "__main__":
    # Define your environment
    # example
    env = make_env(render_mode="rgb_array",enable_wind=True)
    #initialize your agent
    #example
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = xxxAgent(state_dim=obs_dim, n_actions=n_actions)
    agent.load_parameter("xx.pt")
    #test the agent
    testcases=[0,2]
    _ = evaluate(
        agent,
        env,
        testcase=testcases
    )
