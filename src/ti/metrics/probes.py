import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _encode_batch(encode_fn, x):
    z = encode_fn(x)
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    return z


@torch.no_grad()
def _make_split_indices(N, test_frac, seed, device):
    if N <= 1:
        return torch.arange(N, device=device), torch.arange(0, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    perm = torch.randperm(N, generator=g, device=device)
    n_test = int(max(1, round(float(N) * float(test_frac))))
    n_test = min(n_test, N - 1)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


def run_linear_probe_any(encode_fn, obs, y, num_classes, probe_cfg, seed, device):
    N = int(obs.shape[0])
    if N < 2:
        return float("nan"), float("nan")

    train_idx, test_idx = _make_split_indices(N, probe_cfg["test_frac"], seed=seed, device=device)

    with torch.no_grad():
        z0 = _encode_batch(encode_fn, obs[train_idx[: min(64, train_idx.numel())]])
        rep_dim = int(z0.shape[1])

    probe = nn.Linear(rep_dim, int(num_classes)).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(probe_cfg["train_steps"]):
        idx = train_idx[
            torch.randint(
                0, train_idx.numel(), (min(probe_cfg["batch_size"], train_idx.numel()),), device=device
            )
        ]
        with torch.no_grad():
            z = _encode_batch(encode_fn, obs[idx])
        logits = probe(z)
        loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    m = min(probe_cfg["eval_samples"], int(test_idx.numel()))
    idx = test_idx[torch.randint(0, test_idx.numel(), (m,), device=device)]
    with torch.no_grad():
        z = _encode_batch(encode_fn, obs[idx])
        logits = probe(z)
        yb = y[idx]
        acc = (logits.argmax(1) == yb).float().mean().item()
        ce = F.cross_entropy(logits, yb).item()
        mi = max(0.0, float(np.log(float(num_classes)) - ce))
    return acc, mi


def run_xy_regression_probe(encode_fn, obs, probe_cfg, seed, device):
    N = int(obs.shape[0])
    if N < 2:
        return float("nan")

    train_idx, test_idx = _make_split_indices(N, probe_cfg["test_frac"], seed=seed, device=device)

    with torch.no_grad():
        z0 = _encode_batch(encode_fn, obs[train_idx[: min(64, train_idx.numel())]])
        rep_dim = int(z0.shape[1])

    head = nn.Linear(rep_dim, 2).to(device)
    opt = optim.Adam(head.parameters(), lr=1e-3)
    target = obs[:, :2]

    for _ in range(probe_cfg["train_steps"]):
        idx = train_idx[
            torch.randint(
                0, train_idx.numel(), (min(probe_cfg["batch_size"], train_idx.numel()),), device=device
            )
        ]
        with torch.no_grad():
            z = _encode_batch(encode_fn, obs[idx])
        pred = head(z)
        loss = F.mse_loss(pred, target[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    m = min(probe_cfg["eval_samples"], int(test_idx.numel()))
    idx = test_idx[torch.randint(0, test_idx.numel(), (m,), device=device)]
    with torch.no_grad():
        z = _encode_batch(encode_fn, obs[idx])
        pred = head(z)
        mse = F.mse_loss(pred, target[idx]).item()
    return mse


def action_probe_from_pairs(encode_fn, s1, s2, a, probe_cfg, seed, n_actions, device):
    N = int(s1.shape[0])
    if N < 2:
        return float("nan")

    train_idx, test_idx = _make_split_indices(N, probe_cfg["test_frac"], seed=seed, device=device)

    with torch.no_grad():
        z1 = _encode_batch(encode_fn, s1[train_idx[: min(64, train_idx.numel())]])
        rep_dim = int(z1.shape[1])

    clf = nn.Linear(2 * rep_dim, int(n_actions)).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-3)

    for _ in range(probe_cfg["train_steps"]):
        idx = train_idx[
            torch.randint(
                0, train_idx.numel(), (min(probe_cfg["batch_size"], train_idx.numel()),), device=device
            )
        ]
        with torch.no_grad():
            zt = _encode_batch(encode_fn, s1[idx])
            zu = _encode_batch(encode_fn, s2[idx])
        logits = clf(torch.cat([zt, zu], dim=1))
        loss = F.cross_entropy(logits, a[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    m = min(probe_cfg["eval_samples"], int(test_idx.numel()))
    idx = test_idx[torch.randint(0, test_idx.numel(), (m,), device=device)]
    with torch.no_grad():
        zt = _encode_batch(encode_fn, s1[idx])
        zu = _encode_batch(encode_fn, s2[idx])
        logits = clf(torch.cat([zt, zu], dim=1))
        acc = (logits.argmax(1) == a[idx]).float().mean().item()
    return acc
