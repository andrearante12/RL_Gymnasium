import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Shared CNN backbone (Nature DQN architecture) with separate actor and critic heads.

    Input:  (B, 4, 84, 84) stacked grayscale frames
    Shared: Conv(32→64→64) → FC(512)
    Actor:  FC(512→256→n_actions*2) → softplus+1 → alpha, beta per action
    Critic: FC(512→256→1) → scalar value
    """

    def __init__(self, in_channels=4, feature_dim=512, n_actions=3):
        super().__init__()

        # Nature DQN CNN: matches Atari/CarRacing literature
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # → (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(),
        )

        # Actor head: outputs alpha and beta for each Beta distribution
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions * 2),
        )

        # Critic head: outputs scalar state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def features(self, x):
        """Shared feature extraction (CNN + FC)."""
        return self.fc(self.cnn(x))

    def forward(self, x):
        """
        Returns (alpha, beta, value).
        alpha, beta: shape (B, n_actions), each > 1 (unimodal Beta)
        value:       shape (B,)
        """
        h = self.features(x)

        actor_out = self.actor(h)
        alpha, beta = actor_out.chunk(2, dim=-1)
        alpha = F.softplus(alpha) + 1.0
        beta  = F.softplus(beta)  + 1.0

        value = self.critic(h).squeeze(-1)
        return alpha, beta, value


# Keep old names as aliases so any external code importing them still works
class CNN_Net(nn.Module):
    """Deprecated: use ActorCritic instead."""
    def __init__(self, *a, **kw): raise RuntimeError("Use ActorCritic instead of CNN_Net")

class PPO_Net(nn.Module):
    """Deprecated: use ActorCritic instead."""
    def __init__(self, *a, **kw): raise RuntimeError("Use ActorCritic instead of PPO_Net")

class Critic_Net(nn.Module):
    """Deprecated: use ActorCritic instead."""
    def __init__(self, *a, **kw): raise RuntimeError("Use ActorCritic instead of Critic_Net")
