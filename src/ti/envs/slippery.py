import torch

from ti.envs import layouts


class SlipperyDelayMaze:
    D = 3
    N_CLASSES = D

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
        self.qptr = torch.zeros((self.num_envs,), device=device, dtype=torch.long)
        self.queue = torch.randint(0, n_actions, (self.num_envs, self.D), device=device, dtype=torch.long)
        self.goal = torch.tensor(goal, device=device, dtype=torch.long)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        idx = torch.randint(0, self.free.shape[0], (env_ids.numel(),), device=self.device)
        self.pos[env_ids] = self.free[idx]
        self.steps[env_ids] = 0
        self.qptr[env_ids] = torch.randint(0, self.D, (env_ids.numel(),), device=self.device)
        self.queue[env_ids] = torch.randint(0, self.n_actions, (env_ids.numel(), self.D), device=self.device)
        return self._get_obs(env_ids)

    def _get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        xy = layouts.pos_norm_from_grid(self.pos[env_ids], self.maze_size)
        ph = layouts.one_hot3(self.qptr[env_ids])
        return torch.cat([xy, ph], dim=1)

    def step(self, a_cmd):
        a_exec = self.queue[torch.arange(self.num_envs, device=self.device), self.qptr]
        self.queue[torch.arange(self.num_envs, device=self.device), self.qptr] = (
            a_cmd.long().clamp(0, self.n_actions - 1)
        )
        self.qptr = (self.qptr + 1) % self.D

        self.pos = layouts.step_pos_with_layout(
            self.pos, a_exec, self.layout, self.maze_size, self.n_actions
        )
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
        p1 = torch.randint(0, self.D, (num_samples,), device=self.device)
        p2 = torch.randint(0, self.D, (num_samples,), device=self.device)
        xy = layouts.pos_norm_from_grid(pos, self.maze_size)
        o1 = torch.cat([xy, layouts.one_hot3(p1)], dim=1)
        o2 = torch.cat([xy, layouts.one_hot3(p2)], dim=1)
        return o1, o2
