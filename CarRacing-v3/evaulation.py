import gymnasium as gym
from agent import CarRaceAgent
import imageio.v2 as imageio
from env import ImageEnv
import pdb
import torch
import numpy as np
#your submission file
from xxx import xxxAgent, make_env


def evaluation(env_id="CarRacing-v3", env=None, agent=None, testcase=[0,2]):
    rets = []
    for ep in range(len(testcase)):
        frames=[]
        obs, info = env.reset(seed=testcase[ep])
        done = False
        ep_ret = 0.0
        while not done:
            frames.append(env.render())
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        print("Test case {}---score:{}".format(ep,ep_ret))
        imageio.mimsave("testcase{}.gif".format(ep), frames, fps=30)
        rets.append(ep_ret)

    env.close()
    return float(np.mean(rets)), float(np.std(rets))



if __name__ == "__main__":
    # Define your environment
    #example:
    env = make_env(render_mode="rgb_array")
    #initialize your agent
    #example:
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape
    agent = xxxAgent(input_shape, n_actions)
    #load your agent parameter if you have
    #example:
    agent.load_parameter("xxx.pt")
    #run test case
    testcase=[0,2]
    avg, std = evaluation(env_id="CarRacing-v3", env=env,agent=agent, testcase=testcase)
    print("Final eval:", avg, "±", std)
