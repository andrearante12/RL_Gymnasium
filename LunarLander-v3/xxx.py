import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym


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


# Policy network (actor + critic with shared body)

class _Policy(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# Agent

class xxxAgent:
    """
    PPO agent for LunarLander-v3 with discrete actions 
    """

    def __init__(self, state_dim=8, n_actions=4, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = _Policy(state_dim, n_actions).to(self.device)
        self.model  = None  # set when loaded from SB3 checkpoint

    def parameters(self):
        return self.policy.parameters()

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def act(self, s, greedy=False, **kwargs):
        """Returns a single integer action."""
        if self.model is not None:
            action, _ = self.model.predict(s, deterministic=greedy)
            return int(action)
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.policy(x)
            if greedy:
                return int(logits.argmax(dim=-1).item())
            return int(Categorical(logits=logits).sample().item())

    def forward_train(self, s_tensor):
        logits, value = self.policy(s_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, s_tensor, actions_tensor):
        logits, value = self.policy(s_tensor)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_tensor), value, dist.entropy()

    def load_parameter(self, file, **kwargs):
        import os
        base     = os.path.splitext(file)[0]    # "xx.pt" → "xx"
        zip_path = base + ".zip"
        if os.path.exists(zip_path):
            from stable_baselines3 import PPO
            self.model = PPO.load(base, device=self.device)
        else:
            state = torch.load(file, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(state)
        self.policy.eval()

    def save(self, file):
        torch.save(self.policy.state_dict(), file)
