import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.optim as optim

from ti.models.cbm import CBM_Dynamics, RewardPredictor


def sample_negatives_from_pool(pool, B, N, device):
    idx = torch.randint(0, pool.shape[0], (B, N), device=device)
    return pool[idx]


def info_nce_loss_scores(pos_score, neg_scores):
    denom = torch.exp(pos_score) + torch.exp(neg_scores).sum(dim=1)
    return -(pos_score - torch.log(denom + 1e-12)).mean()


def reg_terms_on_scores_and_grad(scores, y):
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=y,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return (scores.pow(2).mean(), grad.pow(2).mean())


@torch.no_grad()
def build_value_pools(buf, obs_dim):
    pools = []
    sp = buf.sp[: buf.size]
    for i in range(obs_dim):
        pools.append(sp[:, i].clone().detach())
    return pools


def train_cbm_models(
    buf,
    obs_dim,
    n_actions,
    maze_size,
    goal,
    steps,
    batch,
    n_neg,
    eps_cmi,
    lam1,
    lam2,
    lr,
    device,
    print_every,
    ckpt_dir=None,
    ckpt_every=None,
    log_dir=None,
    losses_flush_every=200,
    use_amp=False,
    amp_dtype="bf16",
    resume=False,
):
    dyn = CBM_Dynamics(dS=obs_dim, n_actions=n_actions, emb=128, hidden=128).to(device)
    rew = RewardPredictor(x_dim=obs_dim + n_actions, hidden=128).to(device)
    opt_g = optim.Adam(dyn.g.parameters(), lr=lr)
    opt_psi = optim.Adam(dyn.psi.parameters(), lr=lr)
    opt_rew = optim.Adam(rew.parameters(), lr=lr)
    pools = build_value_pools(buf, obs_dim)
    goal = torch.tensor(goal, device=device, dtype=torch.long)

    def compute_reward(sp):
        xy = sp[:, :2]
        pos = torch.round(((xy + 1.0) * 0.5) * float(maze_size - 1)).long().clamp(0, maze_size - 1)
        reached = (pos == goal.unsqueeze(0)).all(dim=1)
        return reached.float()

    g_logger = None
    psi_logger = None
    rew_logger = None
    if log_dir:
        from ti.training.logging import LossLogger
        import os

        g_logger = LossLogger(os.path.join(log_dir, "cbm_g_loss.csv"), flush_every=losses_flush_every)
        psi_logger = LossLogger(os.path.join(log_dir, "cbm_psi_loss.csv"), flush_every=losses_flush_every)
        rew_logger = LossLogger(os.path.join(log_dir, "cbm_rew_loss.csv"), flush_every=losses_flush_every)

    use_amp = bool(use_amp) and device.type == "cuda"
    amp_dtype = str(amp_dtype).lower()
    if amp_dtype in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif amp_dtype in ("fp16", "float16"):
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
    scaler = None
    if use_amp and dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
        if use_amp
        else nullcontext()
    )

    start_g = 0
    if resume and ckpt_dir:
        from ti.training.checkpoint import find_latest_checkpoint, load_checkpoint

        ckpt_path, step = find_latest_checkpoint(ckpt_dir, prefix="cbm_g_step")
        if ckpt_path:
            payload = load_checkpoint(ckpt_path, model=dyn, optimizer=opt_g, map_location=device)
            start_g = int(payload.get("step", step))

    print("    CBM: train g...", flush=True)
    for t in range(start_g, steps // 2):
        s, a, sp, _ = buf.sample(batch)
        with autocast_ctx:
            x = dyn.make_x(s, a)
        for i in range(obs_dim):
            y_true = sp[:, i]
            y_neg = sample_negatives_from_pool(pools[i], batch, n_neg, device)
            with autocast_ctx:
                pos = dyn.g[i].score(y_true, x)
                neg = dyn.g[i].score(
                    y_neg.reshape(-1), x.repeat_interleave(n_neg, dim=0)
                ).reshape(batch, n_neg)
                loss_nce = info_nce_loss_scores(pos, neg)
                y_req = y_true.detach().clone().requires_grad_(True)
                pos_req = dyn.g[i].score(y_req, x)
                l2_s, l2_g = reg_terms_on_scores_and_grad(pos_req, y_req)
                loss = loss_nce + lam1 * l2_s + lam2 * l2_g
            opt_g.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt_g)
                scaler.update()
            else:
                loss.backward()
                opt_g.step()
            if g_logger is not None:
                g_logger.log(t + 1, float(loss.item()))
        if print_every and ((t + 1) % print_every == 0):
            print(f"      g step {t+1:>6}/{steps//2}", flush=True)
        if ckpt_dir and ckpt_every and ((t + 1) % int(ckpt_every) == 0):
            from ti.training.checkpoint import save_checkpoint
            import os

            save_checkpoint(
                os.path.join(ckpt_dir, f"cbm_g_step{t+1:06d}.pt"),
                dyn,
                optimizer=opt_g,
                step=t + 1,
                extra={"component": "g"},
            )

    start_psi = 0
    if resume and ckpt_dir:
        from ti.training.checkpoint import find_latest_checkpoint, load_checkpoint

        ckpt_path, step = find_latest_checkpoint(ckpt_dir, prefix="cbm_psi_step")
        if ckpt_path:
            payload = load_checkpoint(ckpt_path, model=dyn, optimizer=opt_psi, map_location=device)
            start_psi = int(payload.get("step", step))

    print("    CBM: train psi...", flush=True)
    for t in range(start_psi, steps // 2):
        s, a, sp, _ = buf.sample(batch)
        with autocast_ctx:
            x_full = dyn.make_x(s, a)
        j = torch.randint(0, obs_dim, (1,), device=device).item()
        x_m = x_full.clone()
        x_m[:, j] = 0.0
        for i in range(obs_dim):
            y_true = sp[:, i]
            y_neg = sample_negatives_from_pool(pools[i], batch, n_neg, device)
            with autocast_ctx:
                pos = dyn.psi[i].score(y_true, x_m)
                neg = dyn.psi[i].score(
                    y_neg.reshape(-1), x_m.repeat_interleave(n_neg, dim=0)
                ).reshape(batch, n_neg)
                loss_nce = info_nce_loss_scores(pos, neg)
                y_req = y_true.detach().clone().requires_grad_(True)
                pos_req = dyn.psi[i].score(y_req, x_m)
                l2_s, l2_g = reg_terms_on_scores_and_grad(pos_req, y_req)
                loss = loss_nce + lam1 * l2_s + lam2 * l2_g
            opt_psi.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt_psi)
                scaler.update()
            else:
                loss.backward()
                opt_psi.step()
            if psi_logger is not None:
                psi_logger.log(t + 1, float(loss.item()))
        if print_every and ((t + 1) % print_every == 0):
            print(f"      psi step {t+1:>6}/{steps//2}", flush=True)
        if ckpt_dir and ckpt_every and ((t + 1) % int(ckpt_every) == 0):
            from ti.training.checkpoint import save_checkpoint
            import os

            save_checkpoint(
                os.path.join(ckpt_dir, f"cbm_psi_step{t+1:06d}.pt"),
                dyn,
                optimizer=opt_psi,
                step=t + 1,
                extra={"component": "psi"},
            )

    start_rew = 0
    if resume and ckpt_dir:
        from ti.training.checkpoint import find_latest_checkpoint, load_checkpoint

        ckpt_path, step = find_latest_checkpoint(ckpt_dir, prefix="cbm_rew_step")
        if ckpt_path:
            payload = load_checkpoint(ckpt_path, model=rew, optimizer=opt_rew, map_location=device)
            start_rew = int(payload.get("step", step))

    print("    CBM: train reward predictor...", flush=True)
    for t in range(start_rew, steps // 2):
        s, a, sp, _ = buf.sample(batch)
        with autocast_ctx:
            x_full = dyn.make_x(s, a)
        r = compute_reward(sp)
        if random.random() < 0.5:
            x_in = x_full
        else:
            j = torch.randint(0, obs_dim, (1,), device=device).item()
            x_in = x_full.clone()
            x_in[:, j] = 0.0
        with autocast_ctx:
            logp = rew.logp_bernoulli(x_in, r)
            loss = -logp.mean()
        opt_rew.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt_rew)
            scaler.update()
        else:
            loss.backward()
            opt_rew.step()
        if rew_logger is not None:
            rew_logger.log(t + 1, float(loss.item()))
        if print_every and ((t + 1) % print_every == 0):
            print(f"      rew step {t+1:>6}/{steps//2}", flush=True)
        if ckpt_dir and ckpt_every and ((t + 1) % int(ckpt_every) == 0):
            from ti.training.checkpoint import save_checkpoint
            import os

            save_checkpoint(
                os.path.join(ckpt_dir, f"cbm_rew_step{t+1:06d}.pt"),
                rew,
                optimizer=opt_rew,
                step=t + 1,
                extra={"component": "rew"},
            )

    print("    CBM: estimate CMI(dynamics) ...", flush=True)

    @torch.no_grad()
    def estimate_cmi_dyn(num_batches=30):
        cmi = torch.zeros((obs_dim + n_actions, obs_dim), device=device)
        counts = torch.zeros_like(cmi)
        for _ in range(num_batches):
            s, a, sp, _ = buf.sample(batch)
            x_full = dyn.make_x(s, a)
            for i in range(obs_dim):
                y_true = sp[:, i]
                y_neg = sample_negatives_from_pool(pools[i], batch, n_neg, device)
                g_pos = dyn.g[i].score(y_true, x_full)
                g_neg = dyn.g[i].score(
                    y_neg.reshape(-1), x_full.repeat_interleave(n_neg, dim=0)
                ).reshape(batch, n_neg)
                for k in range(obs_dim + n_actions):
                    x_m = x_full.clone()
                    x_m[:, k] = 0.0
                    psi_pos = dyn.psi[i].score(y_true, x_m)
                    psi_neg = dyn.psi[i].score(
                        y_neg.reshape(-1), x_m.repeat_interleave(n_neg, dim=0)
                    ).reshape(batch, n_neg)
                    w = F.softmax(psi_neg, dim=1)
                    phi_pos = g_pos - psi_pos
                    phi_neg = g_neg - psi_neg
                    num = (n_neg + 1.0) * torch.exp(phi_pos)
                    den = torch.exp(phi_pos) + float(n_neg) * (w * torch.exp(phi_neg)).sum(dim=1)
                    cmi_ik = torch.log((num / (den + 1e-12)) + 1e-12)
                    cmi[k, i] += cmi_ik.mean()
                    counts[k, i] += 1.0
        return cmi / (counts + 1e-12)

    print("    CBM: estimate CMI(reward) ...", flush=True)

    @torch.no_grad()
    def estimate_cmi_reward(num_batches=60):
        cmi_r = torch.zeros((obs_dim + n_actions,), device=device)
        counts = torch.zeros_like(cmi_r)
        for _ in range(num_batches):
            s, a, sp, _ = buf.sample(batch)
            x_full = dyn.make_x(s, a)
            r = compute_reward(sp)
            logp_full = rew.logp_bernoulli(x_full, r)
            for k in range(obs_dim + n_actions):
                x_m = x_full.clone()
                x_m[:, k] = 0.0
                logp_m = rew.logp_bernoulli(x_m, r)
                cmi_r[k] += (logp_full - logp_m).mean()
                counts[k] += 1.0
        return cmi_r / (counts + 1e-12)

    cmi_dyn = estimate_cmi_dyn(num_batches=30)
    cmi_rew = estimate_cmi_reward(num_batches=60)
    G_dyn = (cmi_dyn >= eps_cmi).bool()
    PR = (cmi_rew >= eps_cmi).bool()
    if g_logger is not None:
        g_logger.flush()
    if psi_logger is not None:
        psi_logger.flush()
    if rew_logger is not None:
        rew_logger.flush()
    return dyn, rew, cmi_dyn, cmi_rew, G_dyn, PR


def ancestors_in_dyn_graph(G_dyn, reward_parent_mask, obs_dim, device):
    A = G_dyn[:obs_dim, :obs_dim]
    reward_state_parents = reward_parent_mask[:obs_dim].clone()
    parents_of = [torch.nonzero(A[:, i]).squeeze(-1).tolist() for i in range(obs_dim)]
    keep = set(torch.nonzero(reward_state_parents).squeeze(-1).tolist())
    queue = list(keep)
    while queue:
        v = queue.pop()
        for p in parents_of[v]:
            if p not in keep:
                keep.add(p)
                queue.append(p)
    keep_mask = torch.zeros(obs_dim, dtype=torch.bool, device=device)
    if len(keep) > 0:
        keep_mask[list(keep)] = True
    return keep_mask
