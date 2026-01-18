import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarEnergy(nn.Module):
    def __init__(self, x_dim, emb=128, hidden=128):
        super().__init__()
        self.fx = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
        )
        self.hy = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
        )

    def score(self, y, x):
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        fx = self.fx(x)
        hy = self.hy(y)
        return (fx * hy).sum(dim=-1)


class CBM_Dynamics(nn.Module):
    def __init__(self, dS, n_actions, emb=128, hidden=128):
        super().__init__()
        self.dS = int(dS)
        self.n_actions = int(n_actions)
        self.x_dim = self.dS + self.n_actions
        self.g = nn.ModuleList([ScalarEnergy(self.x_dim, emb=emb, hidden=hidden) for _ in range(self.dS)])
        self.psi = nn.ModuleList([ScalarEnergy(self.x_dim, emb=emb, hidden=hidden) for _ in range(self.dS)])

    def make_x(self, s, a):
        a1 = F.one_hot(a, num_classes=self.n_actions).float()
        return torch.cat([s, a1], dim=1)


class RewardPredictor(nn.Module):
    def __init__(self, x_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def logp_bernoulli(self, x, r01):
        logits = self.net(x).squeeze(-1)
        return -F.binary_cross_entropy_with_logits(logits, r01.float(), reduction="none")
