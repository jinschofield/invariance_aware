import torch

from ti.envs import layouts


class TeacupMaze:
    P = 4
    N_CLASSES = P

    def __init__(self, num_envs, maze_size, max_ep_steps, n_actions, goal, device):
        self.num_envs = int(num_envs)
        self.maze_size = int(maze_size)
        self.max_ep_steps = int(max_ep_steps)
        self.n_actions = int(n_actions)
        self.device = device

        self.layout = layouts.make_open_plate_layout(self.maze_size, device)
        self.pos = torch.zeros((self.num_envs, 2), device=device, dtype=torch.long)
        self.steps = torch.zeros((self.num_envs,), device=device, dtype=torch.long)
        self.free = torch.nonzero(~self.layout).long()

        self.centers = torch.tensor([[3, 3], [3, 8], [8, 3], [8, 8], [6, 6]], device=device, dtype=torch.long)
        self.rad2 = 2
        self.phase = torch.zeros((self.num_envs,), device=device, dtype=torch.long)
        self.cup_cells = self._compute_cup_cells()

        self.goal = torch.tensor(goal, device=device, dtype=torch.long)
        self.spin_deltas = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], device=device, dtype=torch.long)
        self.tetra4 = layouts.TETRA4.to(device)

    def _compute_cup_cells(self):
        free = self.free
        d2 = ((free[:, None, :] - self.centers[None, :, :]).float().pow(2)).sum(dim=-1)
        in_cup = d2.min(dim=1).values <= float(self.rad2)
        return free[in_cup]

    def _which_cup(self, pos_xy):
        d2 = ((pos_xy[:, None, :] - self.centers[None, :, :]).float().pow(2)).sum(dim=-1)
        min_d2, cid = d2.min(dim=1)
        in_cup = min_d2 <= float(self.rad2)
        return in_cup, cid

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
        pos = self.pos[env_ids]
        xy = layouts.pos_norm_from_grid(pos, self.maze_size)
        in_cup, cid = self._which_cup(pos)
        local_phase = (self.phase[env_ids] + cid) % self.P
        feat = self.tetra4[local_phase] * in_cup.float().unsqueeze(1)
        return torch.cat([xy, feat], dim=1)

    def step(self, a):
        pos1 = layouts.step_pos_with_layout(
            self.pos, a, self.layout, self.maze_size, self.n_actions
        )
        in_cup, cid = self._which_cup(pos1)
        local_phase = (self.phase + cid) % self.P
        drift = self.spin_deltas[local_phase] * in_cup.long().unsqueeze(1)
        pos2 = pos1 + drift
        pos2[:, 0] = pos2[:, 0].clamp(0, self.maze_size - 1)
        pos2[:, 1] = pos2[:, 1].clamp(0, self.maze_size - 1)
        blocked = self.layout[pos2[:, 0], pos2[:, 1]]
        self.pos = torch.where(blocked.unsqueeze(1), pos1, pos2)

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
    def current_special(self):
        in_cup, _ = self._which_cup(self.pos)
        return in_cup

    @torch.no_grad()
    def current_nuis(self):
        in_cup, cid = self._which_cup(self.pos)
        return (self.phase + cid) % self.P

    @torch.no_grad()
    def sample_invariance_pairs(self, num_samples=2048):
        idx = torch.randint(0, self.cup_cells.shape[0], (num_samples,), device=self.device)
        pos = self.cup_cells[idx]
        p1 = torch.randint(0, self.P, (num_samples,), device=self.device)
        p2 = torch.randint(0, self.P, (num_samples,), device=self.device)
        xy = layouts.pos_norm_from_grid(pos, self.maze_size)
        o1 = torch.cat([xy, self.tetra4[p1]], dim=1)
        o2 = torch.cat([xy, self.tetra4[p2]], dim=1)
        return o1, o2
