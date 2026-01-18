import torch

from ti.envs import layouts


class PeriodicMaze:
    P = 8
    N_CLASSES = P

    def __init__(self, num_envs, maze_size, max_ep_steps, n_actions, goal, device):
        self.num_envs = int(num_envs)
        self.maze_size = int(maze_size)
        self.max_ep_steps = int(max_ep_steps)
        self.n_actions = int(n_actions)
        self.device = device

        self.layout = layouts.make_layout(self.maze_size, device)
        self.pos = torch.zeros((self.num_envs, 2), device=device, dtype=torch.long)
        self.steps = torch.zeros((self.num_envs,), device=device, dtype=torch.long)
        self.free = torch.nonzero(~self.layout).long()
        self.phase = torch.zeros((self.num_envs,), device=device, dtype=torch.long)
        self.goal = torch.tensor(goal, device=device, dtype=torch.long)
        self.tetra4 = layouts.TETRA4.to(device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        idx = torch.randint(0, self.free.shape[0], (env_ids.numel(),), device=self.device)
        self.pos[env_ids] = self.free[idx]
        self.steps[env_ids] = 0
        self.phase[env_ids] = torch.randint(0, self.P, (env_ids.numel(),), device=self.device)
        return self._get_obs(env_ids)

    def _get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        xy = layouts.pos_norm_from_grid(self.pos[env_ids], self.maze_size)
        ph = layouts.phase_sincos3(self.phase[env_ids], self.P)
        return torch.cat([xy, ph], dim=1)

    def step(self, a):
        self.pos = layouts.step_pos_with_layout(
            self.pos, a, self.layout, self.maze_size, self.n_actions
        )
        self.phase = (self.phase + 1) % self.P
        self.steps += 1
        reached = (self.pos == self.goal.unsqueeze(0)).all(dim=1)
        timeouts = self.steps >= self.max_ep_steps
        done = reached | timeouts
        next_obs = self._get_obs()
        reset_obs = next_obs.clone()
        if done.any():
            ids = torch.nonzero(done).squeeze(-1)
            self.reset(ids)
            reset_obs[ids] = self._get_obs(ids)
        return next_obs, done, reset_obs

    @torch.no_grad()
    def sample_invariance_pairs(self, num_samples=2048):
        idx = torch.randint(0, self.free.shape[0], (num_samples,), device=self.device)
        pos = self.free[idx]
        p1 = torch.randint(0, self.P, (num_samples,), device=self.device)
        p2 = torch.randint(0, self.P, (num_samples,), device=self.device)
        xy = layouts.pos_norm_from_grid(pos, self.maze_size)
        o1 = torch.cat([xy, layouts.phase_sincos3(p1, self.P)], dim=1)
        o2 = torch.cat([xy, layouts.phase_sincos3(p2, self.P)], dim=1)
        return o1, o2
