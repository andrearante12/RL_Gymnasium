import gymnasium as gym
from agent import CarRaceAgent
import imageio.v2 as imageio
from env import ImageEnv
import pdb
import torch
import numpy as np
from utils import *


class DiscretizeCarRacing(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # A small but effective discrete set.
        # You can expand this set if you want higher performance.
        self._actions = [
            np.array([ 0.0, 0.0, 0.0], dtype=np.float32),  # do nothing (coast)
            np.array([ 0.0, 1.0, 0.0], dtype=np.float32),  # straight gas
            np.array([ 0.0, 0.0, 0.8], dtype=np.float32),  # brake
            np.array([-1.0, 0.6, 0.0], dtype=np.float32),  # hard left + gas
            np.array([ 1.0, 0.6, 0.0], dtype=np.float32),  # hard right + gas
            np.array([-0.5, 0.8, 0.0], dtype=np.float32),  # gentle left + gas
            np.array([ 0.5, 0.8, 0.0], dtype=np.float32),  # gentle right + gas
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # hard left coast
            np.array([ 1.0, 0.0, 0.0], dtype=np.float32),  # hard right coast
        ]

        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, act: int) -> np.ndarray:
        return self._actions[int(act)].copy()


class FrameStack84(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k, 84, 84),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = preprocess_frame(obs)
        for _ in range(self.k):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = preprocess_frame(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

def make_env(render_mode=None, domain_randomize=False):
    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        domain_randomize=domain_randomize,
        continuous=True,  
    )
    env = DiscretizeCarRacing(env)   
    env = FrameStack84(env, k=4)    
    return env


def calculatescore(returns,steers,env_v,alpha):
    #zigzag penalty
    m = zigzag_metrics(steers, deadband=0.5)
    #halt penalty
    halt=is_halted_speed(env_v)

    return returns-alpha*halt-alpha*m


def evaluate(qfile: str,max_steps: int = 1200, render: bool = True):
    
    seeds=[0,2]
    steers=[]
    device= "cuda" if torch.cuda.is_available() else "cpu"
    #test case 1:
    frames = []
    env = make_env(render_mode="rgb_array",domain_randomize=True)
    #Todo: set your own agent
    agent = CarRaceAgent()
    agent.load_parameter(qfile)
    rets = []
    s, _ = env.reset(seed=seeds[0])
    ep_ret = 0.0
    while True:
        frames.append(env.render())
        #Todo: set the action selection function
        a = agent.act(s)
        steers.append(a)
        s, r, terminated, truncated, info = env.step(a)
        ep_ret += r
        if terminated or truncated:
            break

    ep_ret1=calculatescore(ep_ret,steers,env,0.1)
    env.close()
    
    imageio.mimsave("testcase1.gif", frames, fps=30)
    print("Test case 1 Eval returns:", ep_ret1)

    #test case 2:
    frames2=[]
    env = make_env(render_mode="rgb_array", domain_randomize=True)
    rets = []
    s, _ = env.reset(seed=seeds[1])
    ep_ret = 0.0
    while True:
        frames2.append(env.render())
        #Todo: set the action selection function
        a = agent.act(s, greedy=True)
        steers.append(a)
        s, r, terminated, truncated, info = env.step(a)
        ep_ret += r
        if terminated or truncated:
            break

    ep_ret2=calculatescore(ep_ret,steers,env,0.1)
    env.close()
    imageio.mimsave("testcase2.gif", frames2, fps=30)
    print("Test case 2 Eval returns:", ep_ret2)

    return rets



if __name__ == "__main__":
    #Todo: set your own model parameter file
    qfile="q.pt"
    evaluate(qfile)
