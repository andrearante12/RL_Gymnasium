import gymnasium as gym
from agent import CarRaceAgent
import imageio.v2 as imageio
from env import ImageEnv
import pdb
import torch
import numpy as np
from utils import *

def calculatescore(returns,steers,env_v,alpha):
    #zigzag penalty
    m = zigzag_metrics(steers, deadband=0.5)
    if(m>10):
        print("car zigzags too much")
    else:
        print("car isn't zigzag too much")

    #U turn penalty
    uturn=uturn_metrics(steers,threshold=20)
    if(uturn!=0):
        print("car has U turn")
    else:
        print("car hasn't U turn")

    #halt penalty
    halt=is_halted_speed(env_v)
    if(halt!=0):
        print("car halts for long time")
    else:
        print("car isn't halt for long time")

    return returns-alpha*halt-alpha*uturn-alpha*m


def evaluate(qfile: str,qtars: str,max_steps: int = 1200, render: bool = True):
    
    seeds=[0,2]
    steers=[]
    device= "cuda" if torch.cuda.is_available() else "cpu"
    #test case 1:
    env = make_env(render_mode="rgb_array", domain_randomize=True)
    agent = CarRaceAgent(n_actions=env.action_space.n, device=device)
    agent.load_parameter(qfile,qtars)
    rets = []
    s, _ = env.reset(seed=seeds[0])
    ep_ret = 0.0
    while True:
        a = agent.act(s, greedy=True)
        steers.append(a[0])
        s, r, terminated, truncated, info = env.step(a)
        ep_ret += r
        if terminated or truncated:
            break

    ep_ret1=calculatescore(ep_ret,steers,env,0.01)
    env.close()
    print("Test case 1 Eval returns:", ep_ret1)

    #test case 2:
    env = make_env(render_mode="rgb_array", domain_randomize=True)
    agent = CarRaceAgent(n_actions=env.action_space.n, device=device)
    agent.load_parameter(qfile,qtars)
    rets = []
    s, _ = env.reset(seed=seeds[1])
    ep_ret = 0.0
    while True:
        a = agent.act(s, greedy=True)
        steers.append(a[0])
        s, r, terminated, truncated, info = env.step(a)
        ep_ret += r
        if terminated or truncated:
            break

    ep_ret2=calculatescore(ep_ret,steers,env,0.01)
    env.close()
    print("Test case 2 Eval returns:", ep_ret2)

    return rets


if __name__ == "__main__":
    parser = parse_arg()
    args = parser.parse_args()
    qfile="q.pt"
    qtars="qtar.pt"
    evaluate(qfile,qtars)
