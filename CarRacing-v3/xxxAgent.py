import gymnasium as gym
import numpy as np
from collections import defaultdict
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PPO_Net,Critic_Net,CNN_Net
#CarRaceAgent
#Single agent
#reward: -0.1 for each step without touching any track tile
#+1000/N, if touched a track tile in this track, N 
#is the total number of tiles in this track
import gymnasium as gym
import numpy as np
from collections import defaultdict,deque
from torch.distributions import Beta
from typing import Optional
import random
from dataclasses import dataclass
import cv2
