import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import gymnasium as gym
import torch
import numpy as np
import imageio.v2 as imageio

#donot change the hyperparameter for the environment
def make_env(render_mode=None, enable_wind=True, gravity=-10.0,
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

#define your own agent
class xxxAgent:
    def __init__(self)

    def act(self, s):

    def load_parameter(self, file: str):

    
