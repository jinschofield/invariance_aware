from typing import Dict, Tuple

import torch

from ti.metrics.elliptical import build_precision_A_from_buffer, elliptical_bonus, feat_from_enc

from periodicity_study.common import build_obs_from_pos_phase, free_positions, maze_cfg_from_config


def _student_t_pvalue(t_val: torch.Tensor, df: int) -> torch.Tensor:
    t_val = t_val.double()
    v = torch.tensor(float(df), device=t_val.device, dtype=t_val.dtype)
    x = v / (v + t_val * t_val)
    if hasattr(torch.special, "betainc"):
        ib = torch.special.betainc(0.5 * v, 0.5, x)
        cdf = torch.where(t_val >= 0, 1.0 - 0.5 * ib, 0.5 * ib)
    else:
        try:
            from scipy.stats import t as student_t
        except Exception as exc:
            raise RuntimeError("Student-t CDF requires torch.special.betainc or scipy.") from exc
        cdf = torch.tensor(student_t.cdf(t_val.detach().cpu().numpy(), df), dtype=t_val.dtype)
        cdf = cdf.to(t_val.device)
    p = 2.0 * (1.0 - cdf)
    return torch.clamp(p, 0.0, 1.0)


def _ttest_ind_equal_var(a: torch.Tensor, b: torch.Tensor) -> float:
    n1 = int(a.numel())
    n2 = int(b.numel())
    if n1 < 2 or n2 < 2:
        return float("nan")
    a = a.double()
    b = b.double()
    mean1 = a.mean()
    mean2 = b.mean()
    var1 = a.var(unbiased=True)
    var2 = b.var(unbiased=True)
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2))
    denom = torch.sqrt(sp2 * (1.0 / n1 + 1.0 / n2) + 1e-12)
    t_val = (mean1 - mean2) / denom
    p = _student_t_pvalue(t_val, n1 + n2 - 2)
    return float(p.item())


def _pearsonr(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    n = int(a.numel())
    if n < 3:
        return float("nan"), float("nan")
    a = a.double()
    b = b.double()
    a_mean = a.mean()
    b_mean = b.mean()
    da = a - a_mean
    db = b - b_mean
    cov = (da * db).sum() / max(1, n - 1)
    std_a = torch.sqrt((da * da).sum() / max(1, n - 1))
    std_b = torch.sqrt((db * db).sum() / max(1, n - 1))
    denom = std_a * std_b + 1e-12
    r = torch.clamp(cov / denom, -0.999999, 0.999999)
    t_val = r * torch.sqrt((n - 2) / (1.0 - r * r))
    p = _student_t_pvalue(t_val, n - 2)
    return float(r.item()), float(p.item())


def rep_invariance_by_position(rep, cfg, device: torch.device) -> torch.Tensor:
    maze_cfg = maze_cfg_from_config(cfg)
    free = free_positions(maze_cfg["maze_size"], device)
    P = int(maze_cfg["periodic_P"])

    # Evaluate invariance across all free positions and all phase pairs.
    pos_rep = free.repeat_interleave(P, dim=0)
    phases = torch.arange(P, device=device).repeat(free.shape[0])
    obs = build_obs_from_pos_phase(pos_rep, phases, maze_cfg["maze_size"], P)

    z = rep.encode(obs).reshape(free.shape[0], P, -1)
    d = torch.cdist(z, z, p=2)
    mask = ~torch.eye(P, device=device, dtype=torch.bool)
    d_off = d[:, mask].reshape(free.shape[0], -1)
    return d_off.mean(dim=1)


def build_bonus_heatmaps(rep, buf, cfg, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    maze_cfg = maze_cfg_from_config(cfg)
    enc_fn = lambda x: rep.encode(x)
    A = build_precision_A_from_buffer(
        buf,
        enc_fn,
        maze_cfg["n_actions"],
        lam=cfg.bonus_lambda,
        max_samples=cfg.bonus_max_samples,
        device=device,
    )
    Ainv = torch.linalg.inv(A)

    # Compute bonus across all free positions and all nuisance phases.
    free = free_positions(maze_cfg["maze_size"], device)
    P = int(maze_cfg["periodic_P"])
    actions = (
        torch.arange(maze_cfg["n_actions"], device=device)
        if cfg.bonus_action_avg
        else torch.zeros((1,), device=device, dtype=torch.long)
    )

    pos_rep = free.repeat_interleave(P, dim=0)
    phases = torch.arange(P, device=device).repeat(free.shape[0])
    obs = build_obs_from_pos_phase(pos_rep, phases, maze_cfg["maze_size"], P)

    obs_rep = obs[:, None, :].expand(pos_rep.shape[0], actions.numel(), maze_cfg["obs_dim"]).reshape(
        -1, maze_cfg["obs_dim"]
    )
    a_rep = actions[None, :].expand(pos_rep.shape[0], actions.numel()).reshape(-1)
    phi = feat_from_enc(enc_fn, obs_rep, a_rep, maze_cfg["n_actions"])
    b = elliptical_bonus(phi, Ainv, beta=float(cfg.bonus_beta))
    b = b.reshape(pos_rep.shape[0], actions.numel()).mean(dim=1)

    b_fp = b.reshape(free.shape[0], P)
    mean = b_fp.mean(dim=1)
    std = b_fp.std(dim=1, unbiased=False)

    heat_mean = torch.full(
        (maze_cfg["maze_size"], maze_cfg["maze_size"]), float("nan"), device=device
    )
    heat_std = torch.full(
        (maze_cfg["maze_size"], maze_cfg["maze_size"]), float("nan"), device=device
    )
    heat_mean[free[:, 0], free[:, 1]] = mean
    heat_std[free[:, 0], free[:, 1]] = std
    return heat_mean, heat_std


def bonus_metrics_from_heatmaps(
    heat_mean: torch.Tensor, heat_std: torch.Tensor
) -> Dict[str, float]:
    free_mask = torch.isfinite(heat_mean)
    mean_vals = heat_mean[free_mask]
    std_vals = heat_std[free_mask]

    within_std_mean = float(std_vals.mean().item())
    between_std = float(mean_vals.std(unbiased=False).item())
    ratio = within_std_mean / (between_std + 1e-8)

    return {
        "within_std_mean": within_std_mean,
        "between_std": between_std,
        "within_over_between": ratio,
    }


def heatmap_similarity_metrics(
    heat_a: torch.Tensor, heat_b: torch.Tensor
) -> Dict[str, float]:
    mask = torch.isfinite(heat_a) & torch.isfinite(heat_b)
    a = heat_a[mask]
    b = heat_b[mask]
    if a.numel() < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "l2_norm": float("nan")}
    r, p = _pearsonr(a, b)
    l2 = float(torch.sqrt(torch.mean((a - b) ** 2)).item())
    return {"pearson_r": r, "pearson_p": p, "l2_norm": l2}


def symmetric_kl_torch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def action_dist_kl_by_position(policy, rep, cfg, device: torch.device) -> torch.Tensor:
    maze_cfg = maze_cfg_from_config(cfg)
    free = free_positions(maze_cfg["maze_size"], device)
    P = maze_cfg["periodic_P"]

    obs_list = []
    for phase in range(P):
        phase_t = torch.full((free.shape[0],), phase, device=device, dtype=torch.long)
        obs_list.append(build_obs_from_pos_phase(free, phase_t, maze_cfg["maze_size"], P))

    with torch.no_grad():
        probs_by_phase = []
        for obs in obs_list:
            z = rep.encode(obs).detach()
            probs = policy.action_probs(z)
            probs_by_phase.append(probs)
        probs_by_phase = torch.stack(probs_by_phase, dim=1)

        p = probs_by_phase.unsqueeze(2)
        q = probs_by_phase.unsqueeze(1)
        sym = symmetric_kl_torch(p, q)
        mask = ~torch.eye(P, device=device, dtype=torch.bool)
        sym_off = sym[:, mask].view(sym.shape[0], -1)
        kls = sym_off.mean(dim=1)

    return kls


def pairwise_ttests(values_by_rep: Dict[str, torch.Tensor]) -> Dict[str, float]:
    keys = list(values_by_rep.keys())
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = values_by_rep[keys[i]]
            b = values_by_rep[keys[j]]
            p = _ttest_ind_equal_var(a, b)
            out[f"{keys[i]}_vs_{keys[j]}"] = p
    return out
