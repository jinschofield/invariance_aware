import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        a = a - a.mean(dim=1, keepdim=True)
        return v + a


class DQNAgent:
    def __init__(self, input_dim, n_actions, hidden_dim, lr, device, double=True):
        self.n_actions = int(n_actions)
        self.device = device
        self.double = bool(double)

        self.q = DuelingQNetwork(input_dim, self.n_actions, hidden_dim=hidden_dim).to(device)
        self.q_target = copy.deepcopy(self.q).to(device)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    def act(self, z, epsilon):
        if torch.rand((), device=z.device) < float(epsilon):
            return torch.randint(0, self.n_actions, (z.shape[0],), device=z.device)
        with torch.no_grad():
            q = self.q(z)
        return q.argmax(dim=-1)

    def update(self, batch, gamma, n_step=1):
        s, a, r, sp, d = batch
        q = self.q(s)
        q_a = q.gather(1, a.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                next_a = self.q(sp).argmax(dim=1)
                q_next = self.q_target(sp).gather(1, next_a.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.q_target(sp).max(dim=1).values
            target = r + (1.0 - d.float()) * (gamma ** int(n_step)) * q_next
        loss = F.mse_loss(q_a, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())
