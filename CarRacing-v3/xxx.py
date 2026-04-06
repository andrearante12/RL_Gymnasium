import gymnasium as gym
import numpy as np
from collections import deque
import cv2
import torch
import torch.nn as nn
from torch.distributions import Beta
from utils import ActorCritic


# ---------------------------------------------------------------------------
# Environment wrapper: grayscale + crop + normalize + frame stack
# ---------------------------------------------------------------------------

class CarRacingWrapper(gym.Wrapper):
    """
    Preprocesses CarRacing observations:
      - Converts RGB (96, 96, 3) -> grayscale (84, 84) [crops bottom dashboard]
      - Normalizes to [0.0, 1.0]
      - Stacks n_stack consecutive frames -> (n_stack, 84, 84) float32

    render() delegates to the base env, returning raw RGB for GIF recording.
    """

    def __init__(self, env, n_stack=4, img_size=84):
        super().__init__(env)
        self.n_stack = n_stack
        self.img_size = img_size
        self.frames = deque(maxlen=n_stack)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(n_stack, img_size, img_size),
            dtype=np.float32,
        )

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (96, 96)
        gray = gray[:84, :]                            # crop dashboard (96 -> 84 rows)
        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0

    def _get_obs(self):
        return np.stack(list(self.frames), axis=0)  # (n_stack, H, W)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess(obs)
        for _ in range(self.n_stack):
            self.frames.append(frame)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        car = self.env.unwrapped.car
        if car is not None:
            # Speed bonus (small)
            speed = np.sqrt(car.hull.linearVelocity[0]**2 + car.hull.linearVelocity[1]**2)
            reward += 0.01 * speed / 100.0

            # Grass penalty (gentle)
            wheels_on_grass = sum(1 for w in car.wheels if len(w.tiles) == 0)
            if wheels_on_grass > 0:
                reward -= 0.1 * (wheels_on_grass / 4.0)

            # Steering smoothness penalty (gentle)
            reward -= 0.02 * abs(car.hull.angularVelocity)

        self.frames.append(self._preprocess(obs))
        return self._get_obs(), reward, terminated, truncated, info


class NegativeRewardTerminator(gym.Wrapper):
    """Terminate episode early if cumulative reward drops below threshold."""

    def __init__(self, env, threshold=-20.0):
        super().__init__(env)
        self.threshold = threshold
        self.cumulative_reward = 0.0

    def reset(self, **kwargs):
        self.cumulative_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cumulative_reward += reward
        if self.cumulative_reward < self.threshold:
            terminated = True
        return obs, reward, terminated, truncated, info


class DiscreteActionWrapper(gym.ActionWrapper):
    """Convert discrete action indices to continuous CarRacing actions."""
    ACTIONS = np.array([
        [-1.0, 0.0, 0.0],   # 0: turn left
        [+1.0, 0.0, 0.0],   # 1: turn right
        [ 0.0, 0.0, 0.8],   # 2: brake
        [ 0.0, 1.0, 0.0],   # 3: accelerate
        [ 0.0, 0.0, 0.0],   # 4: do nothing
    ], dtype=np.float32)

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))

    def action(self, act):
        return self.ACTIONS[act]


def make_env(render_mode=None, training=False):
    """Create and wrap CarRacing environment."""
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode, domain_randomize=True)
    env = CarRacingWrapper(env, n_stack=4, img_size=84)
    if training:
        env = NegativeRewardTerminator(env, threshold=-50.0)
    return env


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class xxxAgent(nn.Module):
    """
    PPO agent for CarRacing-v3 using a shared CNN backbone and Beta distribution.

    Architecture: ActorCritic (shared CNN → actor head + critic head)
    Action space: [steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]
      - Gas/brake conflict prevented: brake zeroed when gas > 0.5
    """

    N_ACTIONS   = 3
    N_STACK     = 4
    IMG_SIZE    = 84
    FEATURE_DIM = 512

    def __init__(self, input_shape=None, n_actions=None, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(
            in_channels=self.N_STACK,
            feature_dim=self.FEATURE_DIM,
            n_actions=self.N_ACTIONS,
        )
        self.to(self.device)
        self.model = None   # set when loaded from SB3 checkpoint

    def _to_tensor(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _to_env_action(sample_np):
        """
        Convert Beta samples [0,1]^3 to env action:
          sample[0] -> steering: [0,1] -> [-1,1]
          sample[1] -> gas:      kept in [0,1]
          sample[2] -> brake:    zeroed if gas > 0.5 to prevent conflict
        """
        action = sample_np.copy()
        action[0] = 2.0 * sample_np[0] - 1.0
        if action[1] > 0.5:
            action[2] = 0.0
        return action

    def act(self, obs, **kwargs):
        """
        Inference forward pass. Returns numpy action array of shape (3,).
        Uses SB3 model if loaded via load_parameter; otherwise uses built-in network.
        Eval mode: deterministic (Beta mean). Train mode: samples.
        """
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=not self.training)
            return DiscreteActionWrapper.ACTIONS[action]
        with torch.no_grad():
            x = self._to_tensor(obs)
            alpha, beta, _ = self.net(x)
            if not self.training:
                samples = alpha / (alpha + beta)   # Beta mean
            else:
                samples = Beta(alpha, beta).sample()
            samples = samples.squeeze(0).cpu().numpy()
        return self._to_env_action(samples)

    def evaluate(self, obs_tensor, samples_tensor):
        """
        Recompute log_probs, values, entropy for stored Beta samples.
        Used during PPO update step.
        """
        alpha, beta, value = self.net(obs_tensor)
        dist     = Beta(alpha, beta)
        log_probs = dist.log_prob(samples_tensor).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)
        return log_probs, value, entropy

    def load_parameter(self, file, **kwargs):
        import os
        base     = os.path.splitext(file)[0]    # "xxx.pt" → "xxx"
        zip_path = base + ".zip"
        if os.path.exists(zip_path):
            from stable_baselines3 import PPO
            self.model = PPO.load(base, device=self.device)
        else:
            state_dict = torch.load(file, map_location=self.device, weights_only=True)
            self.load_state_dict(state_dict)
        self.eval()
