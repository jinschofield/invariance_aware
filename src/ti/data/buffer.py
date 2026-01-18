import torch


class EpisodeIndex:
    def __init__(self, starts, lengths, num_envs):
        self.starts = starts
        self.lengths = lengths
        self.num_envs = int(num_envs)
        self.num_episodes = int(starts.numel())


@torch.no_grad()
def build_episode_index_strided(timestep, size, num_envs, device):
    starts, lengths = [], []
    nenv = int(num_envs)
    for e in range(nenv):
        ts = timestep[e:size:nenv]
        if ts.numel() == 0:
            continue
        start_pos = torch.nonzero(ts == 0).squeeze(-1)
        if start_pos.numel() == 0 or int(start_pos[0].item()) != 0:
            start_pos = torch.cat(
                [torch.tensor([0], device=device, dtype=torch.long), start_pos], dim=0
            )
        for j in range(start_pos.numel()):
            sp = int(start_pos[j].item())
            ep_end = int(start_pos[j + 1].item()) if (j + 1) < start_pos.numel() else int(ts.shape[0])
            length = ep_end - sp
            if length >= 1:
                starts.append(e + sp * nenv)
                lengths.append(length)
    return EpisodeIndex(
        torch.tensor(starts, device=device, dtype=torch.long),
        torch.tensor(lengths, device=device, dtype=torch.long),
        nenv,
    )


@torch.no_grad()
def sample_crtr_pairs_offline(buf, epi, batch_size, repetition_factor, k_cap, geom_p, device):
    base_n = batch_size // repetition_factor
    if base_n < 1:
        raise ValueError("batch_size must be >= repetition_factor")
    base_eps = torch.randint(0, epi.num_episodes, (base_n,), device=device)
    ep_ids = base_eps.repeat_interleave(repetition_factor)
    starts = epi.starts[ep_ids]
    lengths = epi.lengths[ep_ids]
    u = (torch.rand(batch_size, device=device) * lengths.float()).long()
    u = torch.minimum(u, lengths - 1)
    geom = torch.distributions.Geometric(probs=torch.tensor(float(geom_p), device=device))
    k = (geom.sample((batch_size,)).long() + 1).clamp(1, int(k_cap))
    uf = u + k
    idx_t = starts + u * epi.num_envs
    idx_f = starts + torch.minimum(uf, lengths - 1) * epi.num_envs
    idx_last = starts + (lengths - 1) * epi.num_envs
    s_t = buf.s[idx_t]
    s_f = buf.s[idx_f]
    overflow = uf >= lengths
    if overflow.any():
        s_f[overflow] = buf.sp[idx_last[overflow]]
    return s_t, s_f


class ReplayBufferPlus:
    def __init__(self, obs_dim, max_size, num_envs, device):
        self.obs_dim = int(obs_dim)
        self.max_size = int(max_size)
        self.num_envs = int(num_envs)
        self.device = device

        self.s = torch.empty((self.max_size, self.obs_dim), device=device)
        self.sp = torch.empty((self.max_size, self.obs_dim), device=device)
        self.a = torch.empty((self.max_size,), device=device, dtype=torch.long)
        self.d = torch.empty((self.max_size,), device=device, dtype=torch.bool)

        self.timestep = torch.empty((self.max_size,), device=device, dtype=torch.long)
        self.current_timestep = torch.zeros((self.num_envs,), device=device, dtype=torch.long)

        self.nuis = torch.empty((self.max_size,), device=device, dtype=torch.long)
        self.special = torch.empty((self.max_size,), device=device, dtype=torch.bool)

        self.size = 0

    def add_batch(self, s, a, sp, done, nuis=None, special=None):
        b = int(s.shape[0])
        end = self.size + b
        if end > self.max_size:
            raise RuntimeError("buffer overflow")
        idx = torch.arange(self.size, end, device=self.device)

        self.s[idx] = s
        self.a[idx] = a
        self.sp[idx] = sp
        self.d[idx] = done

        self.timestep[idx] = self.current_timestep
        self.current_timestep = self.current_timestep + 1
        self.current_timestep[done] = 0

        self.nuis[idx] = 0 if nuis is None else nuis.long()
        self.special[idx] = False if special is None else special.bool()

        self.size = end

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.sp[idx], self.d[idx]
