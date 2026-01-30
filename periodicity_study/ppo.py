from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.utils import seed_everything



class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head_ext = nn.Linear(hidden_dim, 1)
        self.value_head_int = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.policy_head(h)
        value_ext = self.value_head_ext(h).squeeze(-1)
        value_int = self.value_head_int(h).squeeze(-1)
        return logits, value_ext, value_int

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


def _obs_to_pos(obs: torch.Tensor, maze_size: int) -> torch.Tensor:
    xy = obs[:, :2]
    pos = torch.round(((xy + 1.0) * 0.5) * float(maze_size - 1)).long()
    return pos.clamp(0, maze_size - 1)


def _update_running_stats(count: int, mean: float, m2: float, x: torch.Tensor) -> Tuple[int, float, float]:
    if x.numel() == 0:
        return count, mean, m2
    x = x.detach().float()
    batch_count = int(x.numel())
    batch_mean = float(x.mean().item())
    batch_m2 = float(((x - batch_mean) ** 2).sum().item())
    if count == 0:
        return batch_count, batch_mean, batch_m2
    delta = batch_mean - mean
    new_count = count + batch_count
    new_mean = mean + delta * batch_count / new_count
    new_m2 = m2 + batch_m2 + delta * delta * count * batch_count / new_count
    return new_count, new_mean, new_m2


def _compute_gae(rewards, values, dones, next_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def train_ppo(
    rep,
    cfg,
    device: torch.device,
    env_ctor,
    maze_cfg,
    policy_obs_fn=None,
    policy_input_dim: Optional[int] = None,
    use_extrinsic: bool = False,
    rep_updater=None,
    rep_buffer=None,
    rep_update_every: Optional[int] = None,
    rep_update_steps: int = 1,
    rep_batch_size: Optional[int] = None,
    rep_warmup_steps: int = 0,
    eval_callback=None,
    eval_every_updates: Optional[int] = None,
    eval_buffer=None,
) -> Tuple[ActorCritic, List[Dict[str, float]], List[Dict[str, float]]]:
    seed_everything(int(cfg.seed), deterministic=True)
    env = env_ctor(
        num_envs=cfg.ppo_num_envs,
        maze_size=maze_cfg["maze_size"],
        max_ep_steps=maze_cfg["max_ep_steps"],
        n_actions=maze_cfg["n_actions"],
        goal=maze_cfg["goal"],
        device=device,
    )

    if policy_input_dim is None:
        policy_input_dim = int(rep.dim)
    model = ActorCritic(policy_input_dim, cfg.n_actions, cfg.ppo_hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ppo_lr)

    bonus = EpisodicEllipticalBonus(
        z_dim=rep.dim,
        n_actions=cfg.n_actions,
        beta=cfg.bonus_beta,
        lam=cfg.bonus_lambda,
        num_envs=cfg.ppo_num_envs,
        device=device,
    )

    steps_per_update = int(cfg.ppo_steps_per_update)
    total_steps = int(cfg.ppo_total_steps)
    batch_size = steps_per_update * cfg.ppo_num_envs
    updates = max(1, total_steps // batch_size)

    obs = env.reset()
    logs: List[Dict[str, float]] = []
    metrics_log: List[Dict[str, float]] = []
    env_steps = 0
    last_rep_loss = float("nan")
    rep_update_every = int(rep_update_every or cfg.online_rep_update_every)
    rep_update_steps = int(rep_update_steps)
    rep_batch_size = int(rep_batch_size or cfg.online_rep_batch_size)
    rep_warmup_steps = int(rep_warmup_steps or cfg.online_rep_warmup_steps)
    if eval_every_updates is None:
        eval_every_updates = int(cfg.online_eval_every_updates)
    eval_buffer = eval_buffer or rep_buffer

    p_succ_hat = 0.0
    int_count = 0
    int_mean = 0.0
    int_m2 = 0.0
    int_eps = float(getattr(cfg, "ppo_int_norm_eps", 1e-8))
    alpha0 = float(getattr(cfg, "ppo_alpha0", 1.0))
    alpha_eta = float(getattr(cfg, "ppo_alpha_eta", 1.0))
    alpha_rho = float(getattr(cfg, "ppo_alpha_rho", 0.05))
    alpha_zero_after_hit = bool(getattr(cfg, "ppo_alpha_zero_after_hit", False))

    goal = torch.tensor(maze_cfg["goal"], device=device, dtype=torch.long)

    for update in range(1, updates + 1):
        obs_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs, policy_input_dim), device=device)
        actions_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device, dtype=torch.long)
        logprobs_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        values_ext_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        values_int_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        rewards_ext_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        rewards_int_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        dones_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        alpha_mask_buf = (
            torch.ones((steps_per_update, cfg.ppo_num_envs), device=device)
            if alpha_zero_after_hit
            else None
        )
        hit_mask = torch.zeros((cfg.ppo_num_envs,), device=device, dtype=torch.bool)

        for t in range(steps_per_update):
            with torch.no_grad():
                rep_obs = rep.encode(obs).detach()
                policy_obs = rep_obs if policy_obs_fn is None else policy_obs_fn(obs, rep_obs)
                logits, values_ext, values_int = model(policy_obs)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            bonus_vals = bonus.compute_and_update(rep_obs, actions)
            rewards_int = bonus_vals

            next_obs, done, reset_obs = env.step(actions)
            if done.any():
                done_ids = torch.nonzero(done).squeeze(-1)
                bonus.reset(done_ids)
            if use_extrinsic:
                pos_next = _obs_to_pos(next_obs, int(maze_cfg["maze_size"]))
                reached = (pos_next == goal.unsqueeze(0)).all(dim=1)
                rewards_ext = reached.float()
            else:
                rewards_ext = torch.zeros_like(rewards_int)

            obs_buf[t] = policy_obs
            actions_buf[t] = actions
            logprobs_buf[t] = logprobs
            values_ext_buf[t] = values_ext
            values_int_buf[t] = values_int
            rewards_ext_buf[t] = rewards_ext
            rewards_int_buf[t] = rewards_int
            dones_buf[t] = done.float()
            if alpha_mask_buf is not None:
                alpha_mask_buf[t] = (~hit_mask).float()
                hit_mask = hit_mask | (rewards_ext > 0)
                hit_mask = torch.where(done, torch.zeros_like(hit_mask), hit_mask)

            if eval_buffer is not None:
                eval_buffer.add_batch(obs, actions, rewards_int, next_obs, done)
            if rep_buffer is not None and rep_buffer is not eval_buffer:
                rep_buffer.add_batch(obs, actions, rewards_int, next_obs, done)
            env_steps += cfg.ppo_num_envs

            if rep_updater is not None and rep_buffer is not None and rep_buffer.size >= rep_batch_size:
                if env_steps >= rep_warmup_steps and (env_steps % rep_update_every == 0):
                    last_rep_loss = rep_updater(rep_buffer, rep_batch_size, rep_update_steps)

            obs = reset_obs

        with torch.no_grad():
            rep_obs = rep.encode(obs).detach()
            policy_obs = rep_obs if policy_obs_fn is None else policy_obs_fn(obs, rep_obs)
            _, next_value_ext, next_value_int = model(policy_obs)

        rewards_int_flat = rewards_int_buf.reshape(-1)
        if int_count > 1:
            int_sigma = float((int_m2 / max(1, int_count - 1)) ** 0.5)
        else:
            int_sigma = float(rewards_int_flat.std(unbiased=False).item())
        int_sigma = max(int_sigma, int_eps)
        rewards_int_norm = rewards_int_buf / float(int_sigma)

        rollout_success = float((rewards_ext_buf > 0).any().item()) if use_extrinsic else 0.0
        if use_extrinsic:
            p_succ_hat = (1.0 - alpha_rho) * p_succ_hat + alpha_rho * rollout_success
            alpha = alpha0 * ((1.0 - p_succ_hat) ** alpha_eta)
        else:
            alpha = 1.0

        success_rate = float("nan")
        if use_extrinsic:
            success_rate = float((rewards_ext_buf > 0).any(dim=0).float().mean().item())

        if alpha_mask_buf is None:
            alpha_t = alpha
        else:
            alpha_t = alpha * alpha_mask_buf

        adv_ext, ret_ext = _compute_gae(
            rewards_ext_buf,
            values_ext_buf,
            dones_buf,
            next_value_ext,
            cfg.ppo_gamma,
            cfg.ppo_gae_lambda,
        )
        adv_int, ret_int = _compute_gae(
            rewards_int_norm,
            values_int_buf,
            dones_buf,
            next_value_int,
            cfg.ppo_gamma,
            cfg.ppo_gae_lambda,
        )

        adv = adv_ext + alpha_t * adv_int
        ret_mix = rewards_ext_buf + alpha_t * rewards_int_norm
        int_count, int_mean, int_m2 = _update_running_stats(
            int_count, int_mean, int_m2, rewards_int_flat
        )

        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        b_obs = obs_buf.reshape(-1, policy_input_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_returns_ext = ret_ext.reshape(-1)
        b_returns_int = ret_int.reshape(-1)
        b_adv = adv.reshape(-1)
        b_values_ext = values_ext_buf.reshape(-1)
        b_values_int = values_int_buf.reshape(-1)

        n_batch = b_obs.shape[0]
        mb_size = min(int(cfg.ppo_minibatch_size), n_batch)

        for _ in range(int(cfg.ppo_epochs)):
            idx = torch.randperm(n_batch, device=device)
            for start in range(0, n_batch, mb_size):
                mb = idx[start : start + mb_size]
                logits, values_ext, values_int = model(b_obs[mb])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb])
                ratio = (new_logprobs - b_logprobs[mb]).exp()

                pg_loss1 = -b_adv[mb] * ratio
                pg_loss2 = -b_adv[mb] * torch.clamp(
                    ratio, 1.0 - cfg.ppo_clip_coef, 1.0 + cfg.ppo_clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_ext = ((values_ext - b_returns_ext[mb]) ** 2).mean()
                v_loss_int = ((values_int - b_returns_int[mb]) ** 2).mean()
                v_loss = v_loss_ext + v_loss_int
                entropy = dist.entropy().mean()

                loss = pg_loss + cfg.ppo_vf_coef * v_loss - cfg.ppo_ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo_max_grad_norm)
                opt.step()

        logs.append(
            {
                "update": update,
                "env_steps": int(env_steps),
                "mean_reward": float(ret_mix.mean().item()),
                "mean_reward_ext": float(rewards_ext_buf.mean().item()),
                "mean_reward_int": float(rewards_int_buf.mean().item()),
                "value_loss": float(v_loss.item()),
                "policy_loss": float(pg_loss.item()),
                "entropy": float(entropy.item()),
                "rep_loss": float(last_rep_loss),
                "alpha": float(alpha),
                "p_succ_hat": float(p_succ_hat),
                "int_sigma": float(int_sigma),
                "success_rate": float(success_rate),
            }
        )

        if eval_callback is not None and (update == 1 or update % eval_every_updates == 0 or update == updates):
            metrics = eval_callback(update, env_steps, model)
            if use_extrinsic:
                metrics["success_rate"] = float(success_rate)
                metrics["p_succ_hat"] = float(p_succ_hat)
            metrics_log.append(metrics)

    return model, logs, metrics_log
