import torch


@torch.no_grad()
def invariance_metric_from_pairs(encode_fn, obs1, obs2):
    z1 = encode_fn(obs1)
    z2 = encode_fn(obs2)
    if z1.ndim != 2:
        z1 = z1.reshape(z1.shape[0], -1)
    if z2.ndim != 2:
        z2 = z2.reshape(z2.shape[0], -1)
    return (z1 - z2).pow(2).sum(dim=1).mean().item()


@torch.no_grad()
def sample_delayed_pairs_for_slippery(buf, delay_steps, num_envs, device, max_pairs=20000):
    nenv = int(num_envs)
    delay = int(delay_steps)
    max_base = buf.size - (delay * nenv)
    base = torch.arange(0, max_base, device=device)
    idx_delay = base + delay * nenv
    good = buf.timestep[idx_delay] == buf.timestep[base] + delay
    base = base[good]
    if base.numel() == 0:
        return None
    if base.numel() > max_pairs:
        sel = torch.randint(0, base.numel(), (max_pairs,), device=device)
        base = base[sel]
    idx_delay = base + delay * nenv
    s_delay = buf.s[idx_delay]
    sp_delay = buf.sp[idx_delay]
    a_label = buf.a[base]
    return s_delay, sp_delay, a_label
