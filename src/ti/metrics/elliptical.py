import math

import numpy as np
import torch
import torch.nn.functional as F

from ti.envs import layouts


def _onehot_a(a, n_actions):
    return F.one_hot(a.long(), num_classes=n_actions).float()


@torch.no_grad()
def feat_from_enc(enc_fn, obs, a, n_actions):
    z = enc_fn(obs)
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    return torch.cat([z, _onehot_a(a, n_actions)], dim=1)


@torch.no_grad()
def build_precision_A_from_buffer(buf, enc_fn, n_actions, lam, max_samples, device):
    n = int(min(buf.size, max_samples))
    idx = torch.randint(0, buf.size, (n,), device=device)
    s = buf.s[idx]
    a = buf.a[idx]
    phi = feat_from_enc(enc_fn, s, a, n_actions)
    d = int(phi.shape[1])
    A = lam * torch.eye(d, device=device)
    A = A + phi.T @ phi
    return A


@torch.no_grad()
def inv_from_A(A):
    return torch.linalg.inv(A)


@torch.no_grad()
def elliptical_bonus(phi, Ainv, beta, eps=1e-12):
    v = (((phi @ Ainv) * phi).sum(dim=1)).clamp_min(eps)
    return beta * torch.sqrt(v)


def _obs_from_pos_periodic(pos_rc, phase, maze_size, P):
    xy = layouts.pos_norm_from_grid(pos_rc, maze_size)
    ang = (2.0 * math.pi / float(P)) * phase.float()
    ph = torch.stack([torch.sin(ang), torch.cos(ang), torch.sin(2.0 * ang)], dim=-1)
    return torch.cat([xy, ph], dim=1)


def _obs_from_pos_slippery(pos_rc, qptr, maze_size):
    xy = layouts.pos_norm_from_grid(pos_rc, maze_size)
    ph = F.one_hot(qptr.long(), num_classes=3).float()
    return torch.cat([xy, ph], dim=1)


def _obs_from_pos_teacup_inside(pos_rc, phase, maze_size, tetra4):
    xy = layouts.pos_norm_from_grid(pos_rc, maze_size)
    feat = tetra4[phase.long()]
    return torch.cat([xy, feat], dim=1)


@torch.no_grad()
def teacup_inside_mask_grid(maze_size, device):
    centers = torch.tensor([[3, 3], [3, 8], [8, 3], [8, 8], [6, 6]], device=device, dtype=torch.long)
    rad2 = 2
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(maze_size, device=device),
            torch.arange(maze_size, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    d2 = ((grid[:, None, :] - centers[None, :, :]).float().pow(2)).sum(dim=-1)
    inside = (d2.min(dim=1).values <= float(rad2)).reshape(maze_size, maze_size)
    return inside


@torch.no_grad()
def compute_heat_and_sensitivity(env_ctor_name, enc_fn, Ainv, maze_cfg, heat_cfg, device):
    layout = layouts.make_layout(maze_cfg["maze_size"], device)
    free = torch.nonzero(~layout).long()
    Fcells = int(free.shape[0])
    heat_mean = torch.full((maze_cfg["maze_size"], maze_cfg["maze_size"]), float("nan"), device=device)
    heat_std = torch.full((maze_cfg["maze_size"], maze_cfg["maze_size"]), float("nan"), device=device)

    actions = (
        torch.arange(maze_cfg["n_actions"], device=device)
        if heat_cfg["action_avg"]
        else torch.zeros((1,), device=device, dtype=torch.long)
    )

    tetra4 = layouts.TETRA4.to(device)

    CHUNK = 512
    for i in range(0, Fcells, CHUNK):
        pos = free[i : i + CHUNK]
        B = int(pos.shape[0])

        K = int(heat_cfg["nuis_samples"])

        if env_ctor_name == "PeriodicMaze":
            P = int(maze_cfg["periodic_P"])
            ph = torch.randint(0, P, (B, K), device=device)
            pos_rep = pos[:, None, :].expand(B, K, 2).reshape(-1, 2)
            ph_rep = ph.reshape(-1)
            obs = _obs_from_pos_periodic(pos_rep, ph_rep, maze_cfg["maze_size"], P)

        elif env_ctor_name == "SlipperyDelayMaze":
            D = int(maze_cfg["slippery_D"])
            q = torch.randint(0, D, (B, K), device=device)
            pos_rep = pos[:, None, :].expand(B, K, 2).reshape(-1, 2)
            q_rep = q.reshape(-1)
            obs = _obs_from_pos_slippery(pos_rep, q_rep, maze_cfg["maze_size"])

        elif env_ctor_name == "TeacupMaze":
            P = int(maze_cfg["teacup_P"])
            ph = torch.randint(0, P, (B, K), device=device)
            pos_rep = pos[:, None, :].expand(B, K, 2).reshape(-1, 2)
            ph_rep = ph.reshape(-1)
            obs = _obs_from_pos_teacup_inside(pos_rep, ph_rep, maze_cfg["maze_size"], tetra4)

        else:
            raise ValueError(env_ctor_name)

        obs_rep = obs[:, None, :].expand(B * K, actions.numel(), maze_cfg["obs_dim"]).reshape(-1, maze_cfg["obs_dim"])
        a_rep = actions[None, :].expand(B * K, actions.numel()).reshape(-1)

        phi = feat_from_enc(enc_fn, obs_rep, a_rep, maze_cfg["n_actions"])
        b = elliptical_bonus(phi, Ainv, beta=heat_cfg["beta"]).reshape(B * K, actions.numel()).mean(dim=1)

        b_bk = b.reshape(B, K)
        mean = b_bk.mean(dim=1)
        std = b_bk.std(dim=1, unbiased=False)

        heat_mean[pos[:, 0], pos[:, 1]] = mean
        heat_std[pos[:, 0], pos[:, 1]] = std

    return heat_mean, heat_std


def nuisance_predictability_from_bonus(env_ctor_name, enc_fn, Ainv, maze_cfg, probe_cfg, heat_cfg, device):
    layout = layouts.make_layout(maze_cfg["maze_size"], device)
    free = torch.nonzero(~layout).long()
    n = int(min(probe_cfg["samples"], free.shape[0] * 64))
    idx = torch.randint(0, free.shape[0], (n,), device=device)
    pos = free[idx]
    tetra4 = layouts.TETRA4.to(device)

    if env_ctor_name == "PeriodicMaze":
        P = int(maze_cfg["periodic_P"])
        y = torch.randint(0, P, (n,), device=device)
        obs = _obs_from_pos_periodic(pos, y, maze_cfg["maze_size"], P)
        C = P
    elif env_ctor_name == "SlipperyDelayMaze":
        D = int(maze_cfg["slippery_D"])
        y = torch.randint(0, D, (n,), device=device)
        obs = _obs_from_pos_slippery(pos, y, maze_cfg["maze_size"])
        C = D
    elif env_ctor_name == "TeacupMaze":
        P = int(maze_cfg["teacup_P"])
        y = torch.randint(0, P, (n,), device=device)
        obs = _obs_from_pos_teacup_inside(pos, y, maze_cfg["maze_size"], tetra4)
        C = P
    else:
        raise ValueError(env_ctor_name)

    with torch.no_grad():
        actions = (
            torch.arange(maze_cfg["n_actions"], device=device)
            if heat_cfg["action_avg"]
            else torch.zeros((1,), device=device, dtype=torch.long)
        )
        obs_rep = obs[:, None, :].expand(n, actions.numel(), maze_cfg["obs_dim"]).reshape(-1, maze_cfg["obs_dim"])
        a_rep = actions[None, :].expand(n, actions.numel()).reshape(-1)
        phi = feat_from_enc(enc_fn, obs_rep, a_rep, maze_cfg["n_actions"])
        b = elliptical_bonus(phi, Ainv, beta=heat_cfg["beta"]).reshape(n, actions.numel()).mean(dim=1)
        x = b.unsqueeze(1)

    train_idx, test_idx = _make_split_indices(n, probe_cfg["test_frac"], seed=probe_cfg["seed"], device=device)

    clf = torch.nn.Linear(1, C).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for _ in range(probe_cfg["steps"]):
        j = train_idx[
            torch.randint(
                0, train_idx.numel(), (min(probe_cfg["batch"], train_idx.numel()),), device=device
            )
        ]
        logits = clf(x[j])
        loss = F.cross_entropy(logits, y[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        m = min(probe_cfg["eval"], int(test_idx.numel()))
        j = test_idx[torch.randint(0, test_idx.numel(), (m,), device=device)]
        logits = clf(x[j])
        yb = y[j]
        acc = (logits.argmax(1) == yb).float().mean().item()
        ce = F.cross_entropy(logits, yb).item()
        mi = max(0.0, float(np.log(float(C)) - ce))
    return acc, mi


@torch.no_grad()
def _make_split_indices(N, test_frac, seed, device):
    if N <= 1:
        return torch.arange(N, device=device), torch.arange(0, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 12345)
    perm = torch.randperm(N, generator=g, device=device)
    n_test = int(max(1, round(float(N) * float(test_frac))))
    n_test = min(n_test, N - 1)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


@torch.no_grad()
def scalar_scores_from_heat(env_ctor_name, heat_mean, heat_std, layout, teacup_inside_mask=None):
    free_mask = (~layout).clone()
    mean_vals = heat_mean[free_mask]
    std_vals = heat_std[free_mask]

    W = float((std_vals.pow(2).mean()).item())
    B = float((mean_vals.var(unbiased=False)).item())

    within_std = float(std_vals.mean().item())
    within_rel = float((std_vals / (mean_vals.abs() + 1e-8)).mean().item())
    orbit_ratio = float(W / (B + 1e-8))

    out = {
        "heat_mean": float(mean_vals.mean().item()),
        "heat_std_mean": within_std,
        "heat_rel_std_mean": within_rel,
        "orbit_ratio_W_over_B": orbit_ratio,
        "W": W,
        "B": B,
    }

    if env_ctor_name == "TeacupMaze" and teacup_inside_mask is not None:
        inside = teacup_inside_mask & free_mask
        outside = (~teacup_inside_mask) & free_mask
        if inside.any() and outside.any():
            in_mean = float(heat_mean[inside].mean().item())
            out_mean = float(heat_mean[outside].mean().item())
            out["cup_contrast_in_minus_out"] = in_mean - out_mean
            out["inside_std_mean"] = float(heat_std[inside].mean().item())
            out["inside_rel_std_mean"] = float(
                (heat_std[inside] / (heat_mean[inside].abs() + 1e-8)).mean().item()
            )
        else:
            out["cup_contrast_in_minus_out"] = float("nan")
            out["inside_std_mean"] = float("nan")
            out["inside_rel_std_mean"] = float("nan")

    return out
