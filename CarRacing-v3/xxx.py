import gymnasium as gym
import numpy as np
from collections import defaultdict
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PPO_Net,Critic_Net,CNN_Net
import gymnasium as gym
import numpy as np
from collections import defaultdict,deque
from torch.distributions import Beta
from typing import Optional
import random
from dataclasses import dataclass
import cv2

def make_env(render_mode=None):
    """Create and wrap CarRacing environment"""
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode,domain_randomize=True)
    #Your wrapper is here:
    
    return env

#Your agent is here
class xxxAgent(nn.Module):
    def __init__(self, **kwargs):

    def act(self, x, **kwargs):

    def load_parameter(self,file, **kwargs):

    #other functions can be extended here

      
