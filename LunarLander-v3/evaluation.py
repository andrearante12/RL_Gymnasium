from tqdm import tqdm
import gymnasium as gym
import torch
import numpy as np
import imageio.v2 as imageio
from LunarAgent import DQNAgent


def make_env(render_mode=None, enable_wind=False, gravity=-10.0,
             wind_power=15.0, turbulence_power=1.5):
    env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
        continuous=False,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    return env

def evaluate(
    qfile: str,
    seeds=(0, 2),
    max_steps: int = 1200,
    render: bool = True,
    wind_test: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    results = []

    # Create an env once just to read dims
    tmp_env = make_env(render_mode=None, enable_wind=False)
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    n_actions = int(tmp_env.action_space.n)
    tmp_env.close()

    #Todo: set your own agent
    agent = LunarLanderAgent()
    agent.load_parameter(qfile)

    for i, seed in enumerate(seeds, start=1):
        env = make_env(render_mode="rgb_array" if render else None, enable_wind=wind_test)
        frames=[]
        s, _ = env.reset(seed=int(seed))
        ep_ret = 0.0
        ep_len = 0


        angle_sum = 0.0
        angvel_sum = 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            #Todo: set your agents' action selection function
            a = agent.act(s)
            s, r, terminated, truncated, info = env.step(a)

            ep_ret += float(r)
            ep_len += 1

            # obs = [x, y, vx, vy, angle, angularVel, leftLeg, rightLeg]
            angle_sum += float(abs(s[4]))
            angvel_sum += float(abs(s[5]))

            if terminated or truncated:
                break

        env.close()

        # A simple "robustness-adjusted" score (optional)
        # Lower angle/angvel is better.
        imageio.mimsave("./sample_testcase/testcase_{}.gif".format(seed), frames, fps=30)
        adjusted = ep_ret - 0.01 * (angle_sum + angvel_sum)

        print(f"[Eval {i}] | score={adjusted:8.2f}")

        results.append(adjusted)

    return results



if __name__ == "__main__":
    qfile = "q.pt"

    # Evaluate
    _ = evaluate(
        qfile=qfile,
        seeds=(0, 2),
        max_steps=1200,
        render=True,
        wind_test=True,   # stress test with wind
    )
